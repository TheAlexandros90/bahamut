from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import locale
import os
import subprocess
import sys
from typing import Any, Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


SeriesStructure = Literal["single", "panel"]
SplitName = Literal["train", "valid", "test"]
AdapterStrategy = Literal["per_target", "multi_output"]
SUPPORTED_METRICS = {"mae", "rmse", "mape", "smape", "wape"}


def _clean_column_list(columns: Sequence[str], name: str) -> list[str]:
    cleaned = [str(column).strip() for column in columns if str(column).strip()]
    if not cleaned:
        raise ValueError(f"{name} must contain at least one column.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{name} contains duplicated columns: {cleaned}")
    return cleaned


def _clean_optional_column_list(columns: Sequence[str] | None, name: str) -> list[str]:
    if columns is None:
        return []
    cleaned = [str(column).strip() for column in columns if str(column).strip()]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{name} contains duplicated columns: {cleaned}")
    return cleaned


def _clean_metric_list(metrics: Sequence[str]) -> list[str]:
    cleaned = [str(metric).strip().lower() for metric in metrics if str(metric).strip()]
    if not cleaned:
        raise ValueError("metrics must contain at least one metric.")

    unsupported = sorted(set(cleaned) - SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"Unsupported metrics: {unsupported}. Supported: {sorted(SUPPORTED_METRICS)}")

    return cleaned


def _to_index_list(indexer: Sequence[Any] | pd.Index | None, name: str) -> list[Any]:
    if indexer is None:
        raise ValueError(f"{name} is required.")

    values = list(indexer)
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    return values


def _optional_index_list(indexer: Sequence[Any] | pd.Index | None) -> list[Any] | None:
    if indexer is None:
        return None

    values = list(indexer)
    return values or None


def _validate_unique_indexer(indexer: list[Any], name: str) -> None:
    duplicated = pd.Index(indexer)[pd.Index(indexer).duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{name} contains duplicated row identifiers: {duplicated[:5]}")


def _nanmean_or_nan(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _dtype_family(dtype: Any) -> str:
    if is_bool_dtype(dtype):
        return "bool"
    if is_numeric_dtype(dtype):
        return "numeric"
    if is_datetime64_any_dtype(dtype):
        return "datetime"
    return str(dtype)


def _fresh_estimator(blueprint: Any) -> Any:
    if isinstance(blueprint, type):
        estimator = blueprint()
    elif callable(blueprint) and not hasattr(blueprint, "fit"):
        estimator = blueprint()
    else:
        estimator = deepcopy(blueprint)

    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError("The estimator must expose fit and predict.")
    return estimator


def _coerce_target_frame(
    target_like: pd.Series | pd.DataFrame | None,
    split_name: str,
) -> pd.DataFrame | None:
    if target_like is None:
        return None
    if isinstance(target_like, pd.Series):
        if target_like.name is None:
            raise ValueError(f"Bahamut split '{split_name}' contains an unnamed target series.")
        return target_like.to_frame()
    if isinstance(target_like, pd.DataFrame):
        return target_like.copy()
    raise TypeError(f"Bahamut split '{split_name}' has an unsupported target type: {type(target_like)!r}")


def _infer_feature_cols_from_bahamut(
    segments: Mapping[str, Any],
    timestamp_col: str | None = None,
    entity_id_col: str | None = None,
) -> list[str]:
    for key in ("X_train", "X_validation", "X_test"):
        frame = segments.get(key)
        if frame is None:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Bahamut key '{key}' must be a pandas DataFrame.")
        excluded = {timestamp_col, entity_id_col}
        return [column for column in frame.columns.tolist() if column not in excluded]
    raise ValueError("Could not infer feature columns from Bahamut segments.")


def _infer_target_cols_from_bahamut(segments: Mapping[str, Any]) -> list[str]:
    for key in ("y_train", "y_validation", "y_test"):
        target_like = segments.get(key)
        if target_like is None:
            continue
        target_frame = _coerce_target_frame(target_like, split_name=key)
        if target_frame is None:
            continue
        return target_frame.columns.tolist()
    raise ValueError("Could not infer target columns from Bahamut segments.")


def _reconstruct_df_from_bahamut(segments: Mapping[str, Any]) -> pd.DataFrame:
    partition_frames: list[pd.DataFrame] = []

    for split_name, x_key, y_key in (
        ("train", "X_train", "y_train"),
        ("valid", "X_validation", "y_validation"),
        ("test", "X_test", "y_test"),
    ):
        x_part = segments.get(x_key)
        if x_part is None:
            continue
        if not isinstance(x_part, pd.DataFrame):
            raise TypeError(f"Bahamut key '{x_key}' must be a pandas DataFrame.")

        part_df = x_part.copy()
        y_frame = _coerce_target_frame(segments.get(y_key), split_name=split_name)
        if y_frame is not None:
            overlap = sorted(set(part_df.columns) & set(y_frame.columns))
            if overlap:
                raise ValueError(f"Bahamut split '{split_name}' has overlapping X/y columns: {overlap}")
            part_df = part_df.join(y_frame)

        partition_frames.append(part_df)

    if not partition_frames:
        raise ValueError("Bahamut segments do not contain any X partition to reconstruct the dataframe.")

    return pd.concat(partition_frames, axis=0, verify_integrity=True).sort_index()


def _patch_cmdstanpy_windows_encoding() -> None:
    if os.name != "nt":
        return

    import cmdstanpy.model as _eden_cmdstan_model
    import cmdstanpy.utils.command as _eden_cmdstan_command

    if getattr(_eden_cmdstan_command, "_eden_windows_patch_applied", False):
        return

    preferred_encoding = locale.getpreferredencoding(False) or "utf-8"

    def _eden_do_command(cmd, cwd=None, *, fd_out=sys.stdout, pbar=None) -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                bufsize=1,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ,
                universal_newlines=True,
                encoding=preferred_encoding,
                errors="replace",
            )

            while proc.poll() is None:
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    if fd_out is not None:
                        fd_out.write(line)
                    if pbar is not None:
                        pbar(line.strip())

            stdout, _ = proc.communicate()
            if stdout:
                if fd_out is not None:
                    fd_out.write(stdout)
                if pbar is not None:
                    pbar(stdout.strip())

            if proc.returncode != 0:
                raise RuntimeError(f"Command {cmd} returned code {proc.returncode}")
        except OSError as exc:
            raise RuntimeError(f"Command {cmd} failed with error {exc}") from exc

    _eden_cmdstan_command.do_command = _eden_do_command
    _eden_cmdstan_command._eden_windows_patch_applied = True
    _eden_cmdstan_model.do_command = _eden_do_command


def _build_prophet_regressor_aliases(columns: Sequence[str]) -> dict[str, str]:
    return {column: f"eden_reg_{index:02d}" for index, column in enumerate(columns)}


@dataclass(slots=True)
class EdenSpec:
    """Forecasting contract for temporal problems already split externally."""

    timestamp_col: str
    target_cols: list[str]
    feature_cols: list[str]

    series_structure: SeriesStructure = "single"
    entity_id_col: str | None = None

    adapter_strategy: AdapterStrategy = "per_target"
    prediction_suffix: str = "_pred"
    interval_coverage: float | None = 0.9
    calibration_split: SplitName | None = "valid"
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse", "wape"])

    known_future_feature_cols: list[str] = field(default_factory=list)
    enforce_future_covariates: bool = False
    enforce_numeric_features: bool = True
    allow_missing_feature_values: bool = False
    strict_feature_schema: bool = True
    allow_new_entities: bool = False

    expected_frequency: str | None = None
    allow_calendar_gaps: bool = True
    max_missing_periods: int | None = None

    drop_na_target: bool = True
    sort_inputs: bool = True
    check_temporal_order: bool = True
    allow_duplicate_timestamps: bool = False

    def __post_init__(self) -> None:
        self.target_cols = _clean_column_list(self.target_cols, name="target_cols")
        self.feature_cols = _clean_column_list(self.feature_cols, name="feature_cols")
        self.metrics = _clean_metric_list(self.metrics)
        self.known_future_feature_cols = _clean_optional_column_list(
            self.known_future_feature_cols,
            name="known_future_feature_cols",
        )

        if not self.timestamp_col:
            raise ValueError("timestamp_col is required.")
        if not self.prediction_suffix:
            raise ValueError("prediction_suffix must be a non-empty string.")
        if self.series_structure == "panel" and not self.entity_id_col:
            raise ValueError("entity_id_col is required when series_structure='panel'.")
        if self.adapter_strategy not in {"per_target", "multi_output"}:
            raise ValueError("adapter_strategy must be 'per_target' or 'multi_output'.")

        feature_target_overlap = sorted(set(self.target_cols) & set(self.feature_cols))
        if feature_target_overlap:
            raise ValueError(f"Targets and features must be disjoint. Overlap: {feature_target_overlap}")

        future_feature_mismatch = sorted(set(self.known_future_feature_cols) - set(self.feature_cols))
        if future_feature_mismatch:
            raise ValueError(
                f"known_future_feature_cols must be a subset of feature_cols. Invalid: {future_feature_mismatch}"
            )

        if self.interval_coverage is not None and not 0 < float(self.interval_coverage) < 1:
            raise ValueError("interval_coverage must be between 0 and 1.")

        if (not self.allow_calendar_gaps or self.max_missing_periods is not None) and self.expected_frequency is None:
            raise ValueError(
                "expected_frequency is required when allow_calendar_gaps=False or max_missing_periods is set."
            )

        reserved_cols = set(self.target_cols) | set(self.feature_cols) | {self.timestamp_col}
        if self.entity_id_col is not None:
            reserved_cols.add(self.entity_id_col)

        prediction_cols = {f"{target}{self.prediction_suffix}" for target in self.target_cols}
        overlap_reserved = sorted(prediction_cols & reserved_cols)
        if overlap_reserved:
            raise ValueError(
                f"prediction_suffix creates column collisions with input schema: {overlap_reserved}"
            )

    @classmethod
    def from_bahamut_segments(
        cls,
        segments: Mapping[str, Any],
        *,
        timestamp_col: str,
        series_structure: SeriesStructure = "single",
        entity_id_col: str | None = None,
        target_cols: Sequence[str] | None = None,
        feature_cols: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "EdenSpec":
        inferred_target_cols = list(target_cols) if target_cols is not None else _infer_target_cols_from_bahamut(segments)
        inferred_feature_cols = (
            list(feature_cols)
            if feature_cols is not None
            else _infer_feature_cols_from_bahamut(
                segments,
                timestamp_col=timestamp_col,
                entity_id_col=entity_id_col,
            )
        )

        return cls(
            timestamp_col=timestamp_col,
            target_cols=inferred_target_cols,
            feature_cols=inferred_feature_cols,
            series_structure=series_structure,
            entity_id_col=entity_id_col,
            **kwargs,
        )


@dataclass(slots=True)
class EdenSplits:
    """External temporal partitions expressed as row indices over the original dataframe."""

    train_idx: list[Any]
    valid_idx: list[Any] | None = None
    test_idx: list[Any] | None = None

    def __post_init__(self) -> None:
        self.train_idx = _to_index_list(self.train_idx, name="train_idx")
        self.valid_idx = _optional_index_list(self.valid_idx)
        self.test_idx = _optional_index_list(self.test_idx)

        _validate_unique_indexer(self.train_idx, "train_idx")
        if self.valid_idx is not None:
            _validate_unique_indexer(self.valid_idx, "valid_idx")
        if self.test_idx is not None:
            _validate_unique_indexer(self.test_idx, "test_idx")

    @classmethod
    def from_bahamut(cls, segments: Mapping[str, Any]) -> "EdenSplits":
        valid_source = segments.get("indices_validation", segments.get("indices_valid"))
        test_source = segments.get("indices_test")

        return cls(
            train_idx=_to_index_list(segments.get("indices_train"), name="indices_train"),
            valid_idx=_optional_index_list(valid_source),
            test_idx=_optional_index_list(test_source),
        )

    def available_splits(self) -> list[SplitName]:
        names: list[SplitName] = ["train"]
        if self.valid_idx is not None:
            names.append("valid")
        if self.test_idx is not None:
            names.append("test")
        return names


@dataclass(slots=True)
class EdenBahamutBundle:
    df: pd.DataFrame
    splits: EdenSplits
    feature_cols: list[str]
    target_cols: list[str]
    summary: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bahamut(
        cls,
        segments: Mapping[str, Any],
        *,
        df: pd.DataFrame | None = None,
        timestamp_col: str | None = None,
        entity_id_col: str | None = None,
        feature_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
    ) -> "EdenBahamutBundle":
        reconstructed_df = df.copy() if df is not None else _reconstruct_df_from_bahamut(segments)
        inferred_feature_cols = (
            list(feature_cols)
            if feature_cols is not None
            else _infer_feature_cols_from_bahamut(
                segments,
                timestamp_col=timestamp_col,
                entity_id_col=entity_id_col,
            )
        )
        inferred_target_cols = list(target_cols) if target_cols is not None else _infer_target_cols_from_bahamut(segments)

        return cls(
            df=reconstructed_df,
            splits=EdenSplits.from_bahamut(segments),
            feature_cols=inferred_feature_cols,
            target_cols=inferred_target_cols,
            summary=segments.get("resumen_segmentos"),
            metadata={
                "original_keys": sorted(segments.keys()),
                "has_validation": segments.get("X_validation") is not None,
                "has_groups": any(key.startswith("groups_") and segments.get(key) is not None for key in segments),
            },
        )


@dataclass(slots=True)
class BacktestConfig:
    """Walk-forward backtesting configuration over a base split already defined externally."""

    base_split: SplitName = "train"
    initial_train_periods: int | None = None
    horizon: int = 1
    step: int = 1
    gap: int = 0
    max_folds: int | None = None
    expanding_window: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self.step < 1:
            raise ValueError("step must be >= 1.")
        if self.gap < 0:
            raise ValueError("gap must be >= 0.")
        if self.initial_train_periods is not None and self.initial_train_periods < 1:
            raise ValueError("initial_train_periods must be >= 1 when provided.")
        if self.max_folds is not None and self.max_folds < 1:
            raise ValueError("max_folds must be >= 1 when provided.")


class ForecastAdapter(ABC):
    """Minimal adapter contract to decouple Eden from concrete modeling APIs."""

    supports_interval: bool = False

    def required_context_cols(self) -> list[str]:
        return []

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "ForecastAdapter":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    def calibrate(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "ForecastAdapter":
        return self

    def predict_interval(
        self,
        x: pd.DataFrame,
        target_names: list[str],
        coverage: float | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("This adapter does not support prediction intervals.")


class SklearnPerTargetAdapter(ForecastAdapter):
    """Train one sklearn-like estimator per target column."""

    def __init__(self, estimator_blueprint: Any):
        self.estimator_blueprint = estimator_blueprint
        self.models_: dict[str, Any] = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "SklearnPerTargetAdapter":
        self.models_ = {}
        for target in target_names:
            estimator = _fresh_estimator(self.estimator_blueprint)
            estimator.fit(x, y[target])
            self.models_[target] = estimator
        return self

    def predict(self, x: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
        predictions = {}
        for target in target_names:
            predictions[target] = np.asarray(self.models_[target].predict(x))
        return pd.DataFrame(predictions, index=x.index)


class SklearnMultiOutputAdapter(ForecastAdapter):
    """Train a single sklearn-like multi-output estimator."""

    def __init__(self, estimator_blueprint: Any):
        self.estimator_blueprint = estimator_blueprint
        self.model_: Any | None = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "SklearnMultiOutputAdapter":
        self.model_ = _fresh_estimator(self.estimator_blueprint)
        self.model_.fit(x, y[target_names])
        return self

    def predict(self, x: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("The multi-output adapter is not fitted yet.")

        raw_pred = np.asarray(self.model_.predict(x))
        if raw_pred.ndim == 1:
            raw_pred = raw_pred.reshape(-1, 1)
        if raw_pred.shape[1] != len(target_names):
            raise ValueError(
                f"Multi-output estimator returned {raw_pred.shape[1]} columns, expected {len(target_names)}."
            )

        return pd.DataFrame(raw_pred, index=x.index, columns=target_names)


class QuantileEnsembleAdapter(ForecastAdapter):
    """Point plus interval forecasting using explicit lower/upper quantile estimators."""

    supports_interval: bool = True

    def __init__(
        self,
        *,
        center_estimator_blueprint: Any,
        lower_estimator_blueprint: Any,
        upper_estimator_blueprint: Any,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ):
        self.center_estimator_blueprint = center_estimator_blueprint
        self.lower_estimator_blueprint = lower_estimator_blueprint
        self.upper_estimator_blueprint = upper_estimator_blueprint
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)

        if not 0 < self.lower_quantile < self.upper_quantile < 1:
            raise ValueError("lower_quantile and upper_quantile must satisfy 0 < lower < upper < 1.")

        self.models_: dict[str, dict[str, Any]] = {}

    @property
    def interval_coverage(self) -> float:
        return float(self.upper_quantile - self.lower_quantile)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "QuantileEnsembleAdapter":
        self.models_ = {}
        for target in target_names:
            self.models_[target] = {
                "center": _fresh_estimator(self.center_estimator_blueprint),
                "lower": _fresh_estimator(self.lower_estimator_blueprint),
                "upper": _fresh_estimator(self.upper_estimator_blueprint),
            }
            self.models_[target]["center"].fit(x, y[target])
            self.models_[target]["lower"].fit(x, y[target])
            self.models_[target]["upper"].fit(x, y[target])
        return self

    def predict(self, x: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
        predictions = {}
        for target in target_names:
            predictions[target] = np.asarray(self.models_[target]["center"].predict(x), dtype=float)
        return pd.DataFrame(predictions, index=x.index)

    def predict_interval(
        self,
        x: pd.DataFrame,
        target_names: list[str],
        coverage: float | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        requested_coverage = self.interval_coverage if coverage is None else float(coverage)
        if abs(requested_coverage - self.interval_coverage) > 1e-9:
            raise ValueError(
                f"This quantile adapter produces coverage={self.interval_coverage:.2f}; requested coverage={requested_coverage:.2f}."
            )

        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target in target_names:
            lower_raw = np.asarray(self.models_[target]["lower"].predict(x), dtype=float)
            upper_raw = np.asarray(self.models_[target]["upper"].predict(x), dtype=float)
            intervals[target] = (np.minimum(lower_raw, upper_raw), np.maximum(lower_raw, upper_raw))
        return intervals


class ProphetPerTargetAdapter(ForecastAdapter):
    """Adapter nativo para Prophet, con soporte single-series o panel por entidad."""

    supports_interval: bool = True

    def __init__(
        self,
        *,
        timestamp_col: str,
        entity_id_col: str | None = None,
        interval_width: float = 0.9,
        yearly_seasonality: str | bool = False,
        weekly_seasonality: str | bool = False,
        daily_seasonality: str | bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        regressor_cols: Sequence[str] | None = None,
    ):
        self.timestamp_col = str(timestamp_col)
        self.entity_id_col = None if entity_id_col is None else str(entity_id_col)
        self.interval_width = float(interval_width)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = str(seasonality_mode)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.regressor_cols = None if regressor_cols is None else list(regressor_cols)

        if not 0 < self.interval_width < 1:
            raise ValueError("interval_width must be between 0 and 1.")

        self.models_: dict[str, dict[Any, Any]] = {}
        self.regressor_cols_: list[str] = []
        self.regressor_aliases_: dict[str, str] = {}

    def required_context_cols(self) -> list[str]:
        cols = [self.timestamp_col]
        if self.entity_id_col is not None:
            cols.append(self.entity_id_col)
        return cols

    def _fresh_prophet(self) -> Any:
        _patch_cmdstanpy_windows_encoding()
        from prophet import Prophet

        return Prophet(
            interval_width=self.interval_width,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            stan_backend="CMDSTANPY",
        )

    def _group_iterator(self, x: pd.DataFrame) -> list[tuple[Any, pd.DataFrame]]:
        if self.entity_id_col is None:
            return [(None, x)]
        return list(x.groupby(self.entity_id_col, sort=False))

    def _resolve_regressor_cols(self, x: pd.DataFrame) -> list[str]:
        if self.regressor_cols is not None:
            return [column for column in self.regressor_cols if column in x.columns]
        excluded = {self.timestamp_col, self.entity_id_col}
        return [column for column in x.columns if column not in excluded]

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "ProphetPerTargetAdapter":
        if self.timestamp_col not in x.columns:
            raise ValueError(
                f"ProphetPerTargetAdapter expects timestamp column '{self.timestamp_col}' inside the adapter frame."
            )
        if self.entity_id_col is not None and self.entity_id_col not in x.columns:
            raise ValueError(
                f"ProphetPerTargetAdapter expects entity column '{self.entity_id_col}' inside the adapter frame."
            )

        x = x.copy()
        x[self.timestamp_col] = pd.to_datetime(x[self.timestamp_col], errors="raise")
        self.regressor_cols_ = self._resolve_regressor_cols(x)
        self.regressor_aliases_ = _build_prophet_regressor_aliases(self.regressor_cols_)
        self.models_ = {}

        for target in target_names:
            target_models: dict[Any, Any] = {}
            for entity, group in self._group_iterator(x):
                prophet_df = group[[self.timestamp_col] + self.regressor_cols_].copy()
                prophet_df = prophet_df.rename(columns={self.timestamp_col: "ds", **self.regressor_aliases_})
                prophet_df["y"] = y.loc[group.index, target].to_numpy(dtype=float)

                model = self._fresh_prophet()
                for regressor in self.regressor_cols_:
                    model.add_regressor(self.regressor_aliases_[regressor])
                model.fit(prophet_df)
                target_models[entity] = model

            self.models_[target] = target_models

        return self

    def predict(self, x: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
        predictions, _ = self._predict_core(x, target_names)
        return predictions

    def predict_interval(
        self,
        x: pd.DataFrame,
        target_names: list[str],
        coverage: float | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        requested_coverage = self.interval_width if coverage is None else float(coverage)
        if abs(requested_coverage - self.interval_width) > 1e-9:
            raise ValueError(
                f"This Prophet adapter was fitted for coverage={self.interval_width:.2f}; requested coverage={requested_coverage:.2f}."
            )

        _, intervals = self._predict_core(x, target_names)
        return intervals

    def _predict_core(
        self,
        x: pd.DataFrame,
        target_names: list[str],
    ) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
        x = x.copy()
        x[self.timestamp_col] = pd.to_datetime(x[self.timestamp_col], errors="raise")

        predictions: dict[str, pd.Series] = {
            target: pd.Series(index=x.index, dtype=float)
            for target in target_names
        }
        lower_bounds: dict[str, pd.Series] = {
            target: pd.Series(index=x.index, dtype=float)
            for target in target_names
        }
        upper_bounds: dict[str, pd.Series] = {
            target: pd.Series(index=x.index, dtype=float)
            for target in target_names
        }

        for target in target_names:
            target_models = self.models_.get(target, {})
            for entity, group in self._group_iterator(x):
                if entity not in target_models:
                    raise ValueError(f"No Prophet model is available for entity={entity!r} and target='{target}'.")

                future = group[[self.timestamp_col] + self.regressor_cols_].copy()
                future = future.rename(columns={self.timestamp_col: "ds", **self.regressor_aliases_})
                forecast = target_models[entity].predict(future)
                predictions[target].loc[group.index] = forecast["yhat"].to_numpy(dtype=float)
                lower_bounds[target].loc[group.index] = forecast["yhat_lower"].to_numpy(dtype=float)
                upper_bounds[target].loc[group.index] = forecast["yhat_upper"].to_numpy(dtype=float)

        pred_df = pd.DataFrame(
            {target: predictions[target].loc[x.index].to_numpy(dtype=float) for target in target_names},
            index=x.index,
        )
        interval_dict = {
            target: (
                lower_bounds[target].loc[x.index].to_numpy(dtype=float),
                upper_bounds[target].loc[x.index].to_numpy(dtype=float),
            )
            for target in target_names
        }
        return pred_df, interval_dict


class ConformalIntervalAdapter(ForecastAdapter):
    """Add split-conformal style intervals on top of any mean forecaster adapter."""

    supports_interval: bool = True

    def __init__(self, base_adapter: ForecastAdapter, coverage: float = 0.9):
        self.base_adapter = base_adapter
        self.coverage = float(coverage)
        self.residual_quantiles_: dict[str, float] = {}

    def required_context_cols(self) -> list[str]:
        return self.base_adapter.required_context_cols()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "ConformalIntervalAdapter":
        self.base_adapter.fit(x, y, target_names)
        return self

    def predict(self, x: pd.DataFrame, target_names: list[str]) -> pd.DataFrame:
        return self.base_adapter.predict(x, target_names)

    def calibrate(self, x: pd.DataFrame, y: pd.DataFrame, target_names: list[str]) -> "ConformalIntervalAdapter":
        pred_df = self.base_adapter.predict(x, target_names)
        self.residual_quantiles_ = {}

        for target in target_names:
            residuals = np.abs(y[target].to_numpy(dtype=float) - pred_df[target].to_numpy(dtype=float))
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size == 0:
                raise ValueError(f"No finite residuals available to calibrate intervals for target '{target}'.")
            self.residual_quantiles_[target] = float(np.quantile(residuals, self.coverage, method="higher"))

        return self

    def predict_interval(
        self,
        x: pd.DataFrame,
        target_names: list[str],
        coverage: float | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        requested_coverage = self.coverage if coverage is None else float(coverage)
        if abs(requested_coverage - self.coverage) > 1e-9:
            raise ValueError(
                f"This conformal adapter was calibrated for coverage={self.coverage:.2f}; requested coverage={requested_coverage:.2f}. Refit or recalibrate for a different coverage."
            )

        if not self.residual_quantiles_:
            raise RuntimeError("Prediction intervals are not calibrated yet.")

        mean_pred = self.base_adapter.predict(x, target_names)
        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target in target_names:
            spread = self.residual_quantiles_[target]
            center = mean_pred[target].to_numpy(dtype=float)
            intervals[target] = (center - spread, center + spread)
        return intervals


class Eden:
    """Forecasting orchestrator for single or panel time series."""

    def __init__(
        self,
        spec: EdenSpec,
        model: Any | None = None,
        adapter: ForecastAdapter | None = None,
    ):
        if model is None and adapter is None:
            raise ValueError("Provide either model or adapter.")
        if model is not None and adapter is not None:
            raise ValueError("Provide model or adapter, but not both simultaneously.")

        self.spec = spec
        self.model_blueprint = None if isinstance(model, ForecastAdapter) else model
        self.adapter_blueprint = adapter if adapter is not None else model if isinstance(model, ForecastAdapter) else None

        self.adapter_: ForecastAdapter | None = None
        self.feature_names_: list[str] = list(spec.feature_cols)
        self.feature_schema_: dict[str, str] = {}
        self.is_fitted_ = False

        self.interval_calibration_source_: str | None = None
        self.interval_calibration_is_in_sample_: bool = False
        self.train_max_timestamp_: pd.Timestamp | None = None
        self.train_max_timestamp_by_entity_: pd.Series | None = None
        self.trained_entities_: set[Any] = set()
        self.bahamut_metadata_: dict[str, Any] | None = None

        self._validate_spec()

    @classmethod
    def build_from_bahamut(
        cls,
        segments: Mapping[str, Any],
        *,
        timestamp_col: str,
        model: Any | None = None,
        adapter: ForecastAdapter | None = None,
        series_structure: SeriesStructure = "single",
        entity_id_col: str | None = None,
        target_cols: Sequence[str] | None = None,
        feature_cols: Sequence[str] | None = None,
        **spec_kwargs: Any,
    ) -> "Eden":
        spec = EdenSpec.from_bahamut_segments(
            segments,
            timestamp_col=timestamp_col,
            series_structure=series_structure,
            entity_id_col=entity_id_col,
            target_cols=target_cols,
            feature_cols=feature_cols,
            **spec_kwargs,
        )
        return cls(spec=spec, model=model, adapter=adapter)

    def fit(self, df: pd.DataFrame, splits: EdenSplits) -> "Eden":
        prepared = self._prepare_df(df, require_targets=True)
        partitions = self._materialize_splits(prepared, splits, require_targets=True)

        if self.spec.check_temporal_order:
            self._validate_split_boundaries(partitions)

        self.adapter_ = self._build_adapter()
        train_df = partitions["train"]
        x_train = self._adapter_frame(train_df)
        y_train = train_df[self.spec.target_cols]
        self.adapter_.fit(x_train, y_train, self.spec.target_cols)

        if self._supports_interval():
            calibration_name, calibration_df = self._resolve_calibration_partition(partitions)
            calibration_x = self._adapter_frame(calibration_df)
            calibration_y = calibration_df[self.spec.target_cols]
            self.adapter_.calibrate(calibration_x, calibration_y, self.spec.target_cols)
            self.interval_calibration_source_ = calibration_name
            self.interval_calibration_is_in_sample_ = calibration_name == "train"
            if self.interval_calibration_is_in_sample_:
                warnings.warn(
                    "Prediction intervals were calibrated on the training split because no "
                    f"'{self.spec.calibration_split}' partition was available. Split-conformal "
                    "calibration assumes held-out residuals, so these intervals will be too "
                    "narrow and will under-cover. Provide a validation split, or set "
                    "spec.interval_coverage=None to drop the intervals.",
                    UserWarning,
                    stacklevel=2,
                )

        self._store_training_context(train_df)
        self.is_fitted_ = True
        return self

    def fit_from_bahamut(
        self,
        segments: Mapping[str, Any],
        *,
        df: pd.DataFrame | None = None,
    ) -> "Eden":
        bundle = self._bundle_from_bahamut(segments, df=df)
        self.bahamut_metadata_ = bundle.metadata
        return self.fit(bundle.df, bundle.splits)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        pred_df = self._prepare_df(df, require_targets=False)
        self._validate_inference_context(pred_df)

        x = self._adapter_frame(pred_df)
        mean_pred = self._require_adapter().predict(x, self.spec.target_cols)

        out = pred_df.copy()
        for target in self.spec.target_cols:
            out[self._prediction_col(target)] = mean_pred[target].to_numpy(dtype=float)
        return out

    def predict_interval(self, df: pd.DataFrame, coverage: float | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if not self._supports_interval():
            raise RuntimeError("The configured adapter does not provide calibrated prediction intervals.")

        pred_df = self.predict(df)
        x = self._adapter_frame(pred_df)
        resolved_coverage = self._resolve_interval_coverage(coverage)
        intervals = self._require_adapter().predict_interval(x, self.spec.target_cols, coverage=resolved_coverage)

        out = pred_df.copy()
        for target in self.spec.target_cols:
            lower, upper = intervals[target]
            out[self._interval_col(target, "lower", resolved_coverage)] = lower
            out[self._interval_col(target, "upper", resolved_coverage)] = upper
        return out

    def predict_with_metrics(self, df: pd.DataFrame, coverage: float | None = None) -> dict[str, Any]:
        self._check_is_fitted()

        pred_df = self.predict_interval(df, coverage=coverage) if self._supports_interval() else self.predict(df)
        metrics = self._metrics_from_prediction_frame(pred_df) if self._has_all_targets(pred_df) else None
        return {
            "predictions": pred_df,
            "metrics": metrics,
            "metrics_available": metrics is not None,
            "intervals_available": self._supports_interval(),
            "interval_coverage": self._resolve_interval_coverage(coverage) if self._supports_interval() else None,
        }

    def predict_split(
        self,
        df: pd.DataFrame,
        splits: EdenSplits,
        split_name: SplitName = "test",
        coverage: float | None = None,
    ) -> pd.DataFrame:
        partitions = self.partitions(df, splits, require_targets=False)
        if split_name not in partitions:
            raise ValueError(f"Split '{split_name}' is not available. Existing: {sorted(partitions)}")
        return self.predict_interval(partitions[split_name], coverage=coverage) if self._supports_interval() else self.predict(partitions[split_name])

    def predict_split_from_bahamut(
        self,
        segments: Mapping[str, Any],
        *,
        split_name: SplitName = "test",
        coverage: float | None = None,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        bundle = self._bundle_from_bahamut(segments, df=df)
        return self.predict_split(bundle.df, bundle.splits, split_name=split_name, coverage=coverage)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        self._check_is_fitted()

        evaluation = self.predict_with_metrics(df)
        if evaluation["metrics"] is None:
            raise ValueError(
                f"Cannot compute metrics because not all targets are present in df: {self.spec.target_cols}"
            )
        return evaluation["metrics"]

    def evaluate_split(self, df: pd.DataFrame, splits: EdenSplits, split_name: SplitName = "test") -> dict[str, float]:
        partitions = self.partitions(df, splits, require_targets=False)
        if split_name not in partitions:
            raise ValueError(f"Split '{split_name}' is not available. Existing: {sorted(partitions)}")
        return self.evaluate(partitions[split_name])

    def performance_report(
        self,
        df: pd.DataFrame,
        splits: EdenSplits,
        include_train: bool = False,
    ) -> pd.DataFrame:
        self._check_is_fitted()

        partitions = self.partitions(df, splits, require_targets=False)
        rows: list[dict[str, Any]] = []

        for split_name in ("train", "valid", "test"):
            if split_name not in partitions:
                continue
            if split_name == "train" and not include_train:
                continue

            evaluation = self.predict_with_metrics(partitions[split_name])
            if evaluation["metrics"] is None:
                continue

            rows.append({"split": split_name, **evaluation["metrics"]})

        return pd.DataFrame(rows)

    def performance_report_from_bahamut(
        self,
        segments: Mapping[str, Any],
        *,
        include_train: bool = False,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        bundle = self._bundle_from_bahamut(segments, df=df)
        return self.performance_report(bundle.df, bundle.splits, include_train=include_train)

    def backtest(
        self,
        df: pd.DataFrame,
        splits: EdenSplits,
        config: BacktestConfig,
    ) -> dict[str, Any]:
        prepared = self._prepare_df(df, require_targets=True)
        partitions = self._materialize_splits(prepared, splits, require_targets=True)

        if config.base_split not in partitions:
            raise ValueError(
                f"Base split '{config.base_split}' is not available. Existing: {sorted(partitions)}"
            )

        base_df = partitions[config.base_split].copy()
        unique_timestamps = pd.Index(base_df[self.spec.timestamp_col].drop_duplicates().sort_values())
        initial_train_periods = config.initial_train_periods or max(3, config.horizon * 2)

        if len(unique_timestamps) < initial_train_periods + config.gap + config.horizon:
            raise ValueError(
                "Not enough temporal periods in the base split to run the requested backtest configuration."
            )

        fold_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        fold_counter = 0

        stop = len(unique_timestamps) - config.gap - config.horizon + 1
        for anchor in range(initial_train_periods, stop, config.step):
            eval_start = anchor + config.gap
            eval_end = eval_start + config.horizon
            eval_timestamps = unique_timestamps[eval_start:eval_end]
            if len(eval_timestamps) < config.horizon:
                break

            if config.expanding_window:
                train_timestamps = unique_timestamps[:anchor]
            else:
                train_timestamps = unique_timestamps[anchor - initial_train_periods : anchor]

            # El bloque de entrenamiento del fold se parte en fit + calibracion
            # cuando el spec pide intervalos: calibrar conformal sobre residuos
            # de entrenamiento produce bandas sistematicamente estrechas.
            fit_timestamps, calibration_timestamps = self._split_fold_calibration(train_timestamps)

            fold_train_df = base_df[base_df[self.spec.timestamp_col].isin(fit_timestamps)].copy()
            fold_eval_df = base_df[base_df[self.spec.timestamp_col].isin(eval_timestamps)].copy()
            fold_calibration_df = (
                base_df[base_df[self.spec.timestamp_col].isin(calibration_timestamps)].copy()
                if calibration_timestamps is not None
                else None
            )

            if fold_train_df.empty or fold_eval_df.empty:
                continue

            fold_counter += 1
            fold_model = self._spawn_unfitted()
            fold_model.fit(
                base_df,
                EdenSplits(
                    train_idx=fold_train_df.index.tolist(),
                    valid_idx=None if fold_calibration_df is None or fold_calibration_df.empty else fold_calibration_df.index.tolist(),
                ),
            )
            fold_bundle = fold_model.predict_with_metrics(fold_eval_df)

            fold_prediction_df = fold_bundle["predictions"].copy()
            fold_prediction_df["fold"] = fold_counter
            fold_prediction_df["fold_train_end"] = train_timestamps[-1]
            fold_prediction_df["fold_eval_start"] = eval_timestamps[0]
            fold_prediction_df["fold_eval_end"] = eval_timestamps[-1]
            prediction_frames.append(fold_prediction_df)

            metric_row = {
                "fold": fold_counter,
                "train_start": train_timestamps[0],
                "train_end": train_timestamps[-1],
                "eval_start": eval_timestamps[0],
                "eval_end": eval_timestamps[-1],
            }
            if fold_bundle["metrics"] is not None:
                metric_row.update(fold_bundle["metrics"])
            fold_rows.append(metric_row)

            if config.max_folds is not None and fold_counter >= config.max_folds:
                break

        if not fold_rows:
            raise ValueError("Backtest produced no folds. Relax the configuration or increase the base split size.")

        fold_metrics = pd.DataFrame(fold_rows)
        summary = fold_metrics.drop(columns=["fold", "train_start", "train_end", "eval_start", "eval_end"]).mean(numeric_only=True).to_dict()

        return {
            "config": config,
            "fold_metrics": fold_metrics,
            "predictions": pd.concat(prediction_frames, ignore_index=False),
            "summary": summary,
        }

    def backtest_from_bahamut(
        self,
        segments: Mapping[str, Any],
        config: BacktestConfig,
        *,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        bundle = self._bundle_from_bahamut(segments, df=df)
        return self.backtest(bundle.df, bundle.splits, config)

    def partitions(
        self,
        df: pd.DataFrame,
        splits: EdenSplits,
        require_targets: bool = True,
    ) -> dict[str, pd.DataFrame]:
        prepared = self._prepare_df(df, require_targets=require_targets)
        return self._materialize_splits(prepared, splits, require_targets=require_targets)

    def partitions_from_bahamut(
        self,
        segments: Mapping[str, Any],
        *,
        require_targets: bool = True,
        df: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        bundle = self._bundle_from_bahamut(segments, df=df)
        return self.partitions(bundle.df, bundle.splits, require_targets=require_targets)

    def prediction_columns(self) -> dict[str, str]:
        return {target: self._prediction_col(target) for target in self.spec.target_cols}

    def diagnostics(self) -> dict[str, Any]:
        self._check_is_fitted()
        return {
            "adapter_class": type(self._require_adapter()).__name__,
            "supports_interval": self._supports_interval(),
            "interval_calibration_source": self.interval_calibration_source_,
            "interval_calibration_is_in_sample": self.interval_calibration_is_in_sample_,
            "trained_targets": list(self.spec.target_cols),
            "feature_schema": dict(self.feature_schema_),
            "known_future_feature_cols": list(self.spec.known_future_feature_cols),
            "adapter_context_cols": self._require_adapter().required_context_cols(),
            "bahamut_source_loaded": self.bahamut_metadata_ is not None,
        }

    def run(
        self,
        df: pd.DataFrame,
        splits: EdenSplits,
        include_train_metrics: bool = False,
    ) -> dict[str, Any]:
        self.fit(df, splits)

        output: dict[str, Any] = {
            "model": self,
            "prediction_columns": self.prediction_columns(),
            "available_splits": splits.available_splits(),
            "diagnostics": self.diagnostics(),
        }

        partitions = self.partitions(df, splits, require_targets=False)

        if include_train_metrics and "train" in partitions:
            train_bundle = self.predict_with_metrics(partitions["train"])
            output["train_predictions"] = train_bundle["predictions"]
            if train_bundle["metrics"] is not None:
                output["train_metrics"] = train_bundle["metrics"]

        for split_name in ("valid", "test"):
            if split_name not in partitions:
                continue

            evaluation = self.predict_with_metrics(partitions[split_name])
            output[f"{split_name}_predictions"] = evaluation["predictions"]
            if evaluation["metrics"] is not None:
                output[f"{split_name}_metrics"] = evaluation["metrics"]

        output["performance_report"] = self.performance_report(
            df,
            splits,
            include_train=include_train_metrics,
        )
        return output

    def run_from_bahamut(
        self,
        segments: Mapping[str, Any],
        *,
        include_train_metrics: bool = False,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        bundle = self._bundle_from_bahamut(segments, df=df)
        self.bahamut_metadata_ = bundle.metadata
        output = self.run(bundle.df, bundle.splits, include_train_metrics=include_train_metrics)
        output["bahamut_feature_cols"] = bundle.feature_cols
        output["bahamut_target_cols"] = bundle.target_cols
        output["bahamut_summary"] = bundle.summary
        output["bahamut_metadata"] = bundle.metadata
        return output

    def _validate_spec(self) -> None:
        if self.adapter_blueprint is None and self.model_blueprint is None:
            raise ValueError("No model blueprint is available to build the forecasting adapter.")

    def _prepare_df(self, df: pd.DataFrame, require_targets: bool) -> pd.DataFrame:
        if df is None:
            raise ValueError("df is None.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df is empty.")

        required_cols = set(self.spec.feature_cols + [self.spec.timestamp_col])
        if require_targets:
            required_cols.update(self.spec.target_cols)
        if self.spec.series_structure == "panel":
            required_cols.add(self.spec.entity_id_col)  # type: ignore[arg-type]

        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"df is missing required columns: {missing}")

        out = df.copy()
        out[self.spec.timestamp_col] = pd.to_datetime(out[self.spec.timestamp_col], errors="raise")

        if self.spec.enforce_numeric_features:
            non_numeric_features = [col for col in self.spec.feature_cols if not is_numeric_dtype(out[col])]
            if non_numeric_features:
                raise ValueError(f"All feature columns must be numeric. Invalid: {non_numeric_features}")

            available_targets = [col for col in self.spec.target_cols if col in out.columns]
            non_numeric_targets = [col for col in available_targets if not is_numeric_dtype(out[col])]
            if non_numeric_targets:
                raise ValueError(f"All target columns must be numeric. Invalid: {non_numeric_targets}")

        if not self.spec.allow_missing_feature_values:
            feature_cols_with_na = [col for col in self.spec.feature_cols if out[col].isna().any()]
            if feature_cols_with_na:
                raise ValueError(f"Feature columns contain missing values: {feature_cols_with_na}")

        if require_targets and self.spec.drop_na_target:
            out = out.dropna(subset=self.spec.target_cols)
            if out.empty:
                raise ValueError("All rows were dropped after removing NA targets.")

        return out

    def _materialize_splits(
        self,
        df: pd.DataFrame,
        splits: EdenSplits,
        require_targets: bool,
    ) -> dict[str, pd.DataFrame]:
        raw_split_map: dict[str, list[Any] | None] = {
            "train": splits.train_idx,
            "valid": splits.valid_idx,
            "test": splits.test_idx,
        }

        partitions: dict[str, pd.DataFrame] = {}
        for split_name, indexer in raw_split_map.items():
            if indexer is None:
                continue

            part_df = self._take_rows(df, indexer, split_name=split_name)
            if require_targets and self.spec.drop_na_target:
                part_df = part_df.dropna(subset=self.spec.target_cols)

            if part_df.empty:
                raise ValueError(f"Split '{split_name}' is empty after preprocessing.")

            if self.spec.sort_inputs:
                part_df = part_df.sort_values(self._sort_columns())

            self._validate_temporal_structure(part_df, split_name=split_name)
            partitions[split_name] = part_df.copy()

        self._validate_disjoint_splits(partitions)
        return partitions

    def _take_rows(self, df: pd.DataFrame, indexer: Sequence[Any], split_name: str) -> pd.DataFrame:
        labels = list(indexer)
        label_index = pd.Index(labels)

        if label_index.isin(df.index).all():
            return df.loc[labels].copy()

        if all(isinstance(value, (int, np.integer)) for value in labels):
            positions = [int(value) for value in labels]
            if min(positions) < 0 or max(positions) >= len(df):
                raise ValueError(f"Split '{split_name}' contains out-of-bounds positional indices.")
            return df.iloc[positions].copy()

        raise ValueError(
            f"Split '{split_name}' could not be resolved against df.index and is not positional."
        )

    def _sort_columns(self) -> list[str]:
        columns: list[str] = []
        if self.spec.series_structure == "panel":
            columns.append(self.spec.entity_id_col)  # type: ignore[arg-type]
        columns.append(self.spec.timestamp_col)
        return columns

    def _validate_temporal_structure(self, df: pd.DataFrame, split_name: str) -> None:
        ts_col = self.spec.timestamp_col

        if self.spec.series_structure == "single":
            if not df[ts_col].is_monotonic_increasing:
                raise ValueError(f"{split_name}: timestamps must be increasing.")

            if not self.spec.allow_duplicate_timestamps and df[ts_col].duplicated().any():
                raise ValueError(f"{split_name}: duplicate timestamps found.")

            self._validate_calendar(df[ts_col], split_name=split_name)
            return

        entity_col = self.spec.entity_id_col
        assert entity_col is not None

        for entity, group in df.groupby(entity_col, sort=False):
            if not group[ts_col].is_monotonic_increasing:
                raise ValueError(
                    f"{split_name}: timestamps must be increasing within each entity. Failed for entity={entity!r}"
                )

            if not self.spec.allow_duplicate_timestamps and group[ts_col].duplicated().any():
                raise ValueError(f"{split_name}: duplicate timestamps found for entity={entity!r}")

            self._validate_calendar(group[ts_col], split_name=split_name, entity_label=entity)

    def _validate_calendar(
        self,
        timestamps: pd.Series,
        split_name: str,
        entity_label: Any | None = None,
    ) -> None:
        if self.spec.expected_frequency is None:
            return

        observed = pd.Index(pd.Series(timestamps).drop_duplicates().sort_values())
        if len(observed) <= 1:
            return

        expected = pd.date_range(start=observed.min(), end=observed.max(), freq=self.spec.expected_frequency)
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        prefix = split_name if entity_label is None else f"{split_name} entity={entity_label!r}"

        if len(unexpected) > 0:
            raise ValueError(
                f"{prefix}: timestamps are not aligned with expected_frequency='{self.spec.expected_frequency}'."
            )

        if not self.spec.allow_calendar_gaps and len(missing) > 0:
            raise ValueError(
                f"{prefix}: calendar gaps found. Missing periods: {list(missing[:5])}"
            )

        if self.spec.max_missing_periods is not None and len(missing) > self.spec.max_missing_periods:
            raise ValueError(
                f"{prefix}: too many missing periods ({len(missing)}), maximum allowed is {self.spec.max_missing_periods}."
            )

    def _validate_disjoint_splits(self, partitions: Mapping[str, pd.DataFrame]) -> None:
        split_names = list(partitions)
        for idx, left_name in enumerate(split_names):
            left_index = set(partitions[left_name].index.tolist())
            for right_name in split_names[idx + 1 :]:
                overlap = left_index & set(partitions[right_name].index.tolist())
                if overlap:
                    preview = sorted(overlap)[:5]
                    raise ValueError(
                        f"Rows are shared between splits '{left_name}' and '{right_name}'. Sample overlap: {preview}"
                    )

    def _validate_split_boundaries(self, partitions: Mapping[str, pd.DataFrame]) -> None:
        ordered_pairs = [("train", "valid"), ("valid", "test"), ("train", "test")]
        for left_name, right_name in ordered_pairs:
            if left_name in partitions and right_name in partitions:
                self._validate_single_boundary(
                    partitions[left_name],
                    partitions[right_name],
                    left_name=left_name,
                    right_name=right_name,
                )

    def _validate_single_boundary(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_name: str,
        right_name: str,
    ) -> None:
        ts_col = self.spec.timestamp_col
        strict = not self.spec.allow_duplicate_timestamps

        if self.spec.series_structure == "single":
            left_max = left_df[ts_col].max()
            right_min = right_df[ts_col].min()

            leakage = left_max >= right_min if strict else left_max > right_min
            if leakage:
                raise ValueError(
                    f"Temporal leakage detected: max({left_name})={left_max} is not earlier than min({right_name})={right_min}."
                )
            return

        entity_col = self.spec.entity_id_col
        assert entity_col is not None

        left_max_by_entity = left_df.groupby(entity_col)[ts_col].max()
        right_min_by_entity = right_df.groupby(entity_col)[ts_col].min()
        shared_entities = left_max_by_entity.index.intersection(right_min_by_entity.index)

        for entity in shared_entities:
            left_max = left_max_by_entity.loc[entity]
            right_min = right_min_by_entity.loc[entity]
            leakage = left_max >= right_min if strict else left_max > right_min
            if leakage:
                raise ValueError(
                    f"Temporal leakage detected for entity={entity!r}: max({left_name})={left_max} is not earlier than min({right_name})={right_min}."
                )

    def _build_adapter(self) -> ForecastAdapter:
        if self.adapter_blueprint is not None:
            adapter = deepcopy(self.adapter_blueprint)
        else:
            if self.spec.adapter_strategy == "multi_output":
                adapter = SklearnMultiOutputAdapter(self.model_blueprint)
            else:
                adapter = SklearnPerTargetAdapter(self.model_blueprint)

        if self.spec.interval_coverage is not None and not adapter.supports_interval:
            adapter = ConformalIntervalAdapter(adapter, coverage=self.spec.interval_coverage)
        return adapter

    def _require_adapter(self) -> ForecastAdapter:
        if self.adapter_ is None:
            raise RuntimeError("The forecasting adapter is not available yet. Fit Eden first.")
        return self.adapter_

    def _adapter_input_cols(self) -> list[str]:
        adapter = self._require_adapter()
        columns = list(self.spec.feature_cols)
        for column in adapter.required_context_cols():
            if column not in columns:
                columns.append(column)
        return columns

    def _adapter_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self._adapter_input_cols()
        missing = sorted(set(columns) - set(df.columns))
        if missing:
            raise ValueError(f"Adapter input requires missing columns: {missing}")
        return df[columns].copy()

    def _supports_interval(self) -> bool:
        return self.adapter_ is not None and self.adapter_.supports_interval

    def _resolve_interval_coverage(self, coverage: float | None) -> float:
        if coverage is not None:
            return float(coverage)
        if self.spec.interval_coverage is None:
            raise ValueError("No interval coverage is configured for this Eden instance.")
        return float(self.spec.interval_coverage)

    def _resolve_calibration_partition(
        self,
        partitions: Mapping[str, pd.DataFrame],
    ) -> tuple[str, pd.DataFrame]:
        preferred = self.spec.calibration_split
        if preferred is not None and preferred in partitions:
            return preferred, partitions[preferred]
        if "valid" in partitions:
            return "valid", partitions["valid"]
        return "train", partitions["train"]

    def _store_training_context(self, train_df: pd.DataFrame) -> None:
        self.feature_schema_ = {column: _dtype_family(train_df[column].dtype) for column in self.spec.feature_cols}
        self.train_max_timestamp_ = pd.Timestamp(train_df[self.spec.timestamp_col].max())

        if self.spec.series_structure == "panel":
            entity_col = self.spec.entity_id_col
            assert entity_col is not None
            self.train_max_timestamp_by_entity_ = train_df.groupby(entity_col)[self.spec.timestamp_col].max()
            self.trained_entities_ = set(train_df[entity_col].unique().tolist())
        else:
            self.train_max_timestamp_by_entity_ = None
            self.trained_entities_ = set()

    def _validate_inference_context(self, df: pd.DataFrame) -> None:
        self._validate_feature_schema(df)
        self._validate_entity_policy(df)
        self._validate_future_covariates(df)

    def _validate_feature_schema(self, df: pd.DataFrame) -> None:
        if not self.spec.strict_feature_schema:
            return

        for column in self.spec.feature_cols:
            expected_family = self.feature_schema_.get(column)
            current_family = _dtype_family(df[column].dtype)
            if expected_family is None:
                continue
            if expected_family != current_family:
                raise ValueError(
                    f"Feature schema mismatch for column '{column}': expected {expected_family}, got {current_family}."
                )

    def _validate_entity_policy(self, df: pd.DataFrame) -> None:
        if self.spec.series_structure != "panel" or self.spec.allow_new_entities:
            return

        entity_col = self.spec.entity_id_col
        assert entity_col is not None
        new_entities = sorted(set(df[entity_col].unique().tolist()) - self.trained_entities_)
        if new_entities:
            raise ValueError(f"Prediction contains unseen entities: {new_entities[:5]}")

    def _validate_future_covariates(self, df: pd.DataFrame) -> None:
        if not self.spec.enforce_future_covariates:
            return

        unknown_future_features = sorted(set(self.spec.feature_cols) - set(self.spec.known_future_feature_cols))
        if not unknown_future_features:
            return

        if self.spec.series_structure == "single":
            assert self.train_max_timestamp_ is not None
            future_mask = df[self.spec.timestamp_col] > self.train_max_timestamp_
        else:
            entity_col = self.spec.entity_id_col
            assert entity_col is not None
            assert self.train_max_timestamp_by_entity_ is not None

            mapped_cutoff = df[entity_col].map(self.train_max_timestamp_by_entity_)
            future_mask = mapped_cutoff.isna() | (df[self.spec.timestamp_col] > mapped_cutoff)

        if future_mask.any():
            raise ValueError(
                "Rows beyond the training cutoff require every feature to be declared in "
                f"known_future_feature_cols. Unknown future features: {unknown_future_features}"
            )

    def _bundle_from_bahamut(
        self,
        segments: Mapping[str, Any],
        *,
        df: pd.DataFrame | None = None,
    ) -> EdenBahamutBundle:
        bundle = EdenBahamutBundle.from_bahamut(
            segments,
            df=df,
            timestamp_col=self.spec.timestamp_col,
            entity_id_col=self.spec.entity_id_col,
            feature_cols=self.spec.feature_cols,
            target_cols=self.spec.target_cols,
        )

        required_context = [self.spec.timestamp_col]
        if self.spec.entity_id_col is not None:
            required_context.append(self.spec.entity_id_col)

        missing_context = [column for column in required_context if column not in bundle.df.columns]
        if missing_context:
            raise ValueError(
                "Bahamut segments do not contain all temporal context columns required by Eden: "
                f"{missing_context}. Include them in Bahamut X_* or pass df=original_dataframe."
            )

        missing_features = [column for column in self.spec.feature_cols if column not in bundle.df.columns]
        if missing_features:
            raise ValueError(f"Bahamut bundle is missing required feature columns: {missing_features}")

        missing_targets = [column for column in self.spec.target_cols if column not in bundle.df.columns]
        if missing_targets:
            raise ValueError(f"Bahamut bundle is missing required target columns: {missing_targets}")

        return bundle

    def _split_fold_calibration(self, train_timestamps: pd.Index) -> tuple[pd.Index, pd.Index | None]:
        """Reserva la cola del bloque de entrenamiento para calibrar intervalos.

        Devuelve `(fit_timestamps, calibration_timestamps)`. Si el spec no pide
        intervalos, o el bloque es demasiado corto para partirlo, la calibracion
        es `None` y se entrena con todo el bloque.
        """

        if self.spec.interval_coverage is None:
            return train_timestamps, None

        total = len(train_timestamps)
        calibration_periods = max(1, int(round(total * 0.2)))
        if total - calibration_periods < 2:
            return train_timestamps, None

        cut = total - calibration_periods
        return train_timestamps[:cut], train_timestamps[cut:]

    def _spawn_unfitted(self) -> "Eden":
        return Eden(
            spec=deepcopy(self.spec),
            model=self.model_blueprint,
            adapter=self.adapter_blueprint,
        )

    def _prediction_col(self, target: str) -> str:
        return f"{target}{self.spec.prediction_suffix}"

    def _interval_col(self, target: str, side: Literal["lower", "upper"], coverage: float) -> str:
        pct = int(round(coverage * 100))
        return f"{target}_{side}_{pct}"

    def _has_all_targets(self, df: pd.DataFrame) -> bool:
        return set(self.spec.target_cols).issubset(df.columns)

    def _metrics_from_prediction_frame(self, pred_df: pd.DataFrame) -> dict[str, float]:
        if not self._has_all_targets(pred_df):
            raise ValueError(f"Not all targets are present in the provided dataframe: {self.spec.target_cols}")

        missing_prediction_cols = [
            self._prediction_col(target)
            for target in self.spec.target_cols
            if self._prediction_col(target) not in pred_df.columns
        ]
        if missing_prediction_cols:
            raise ValueError(f"Prediction columns are missing: {missing_prediction_cols}")

        metric_df = pred_df.dropna(subset=self.spec.target_cols)
        if metric_df.empty:
            raise ValueError("No rows with non-null targets are available to compute metrics.")

        results: dict[str, float] = {}
        multi_target = len(self.spec.target_cols) > 1

        for target in self.spec.target_cols:
            y_true = metric_df[target].to_numpy(dtype=float)
            y_pred = metric_df[self._prediction_col(target)].to_numpy(dtype=float)
            metrics = self._compute_metrics(y_true=y_true, y_pred=y_pred)

            if multi_target:
                for metric_name, value in metrics.items():
                    results[f"{target}__{metric_name}"] = value
            else:
                results.update(metrics)

        return results

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        results: dict[str, float] = {}

        for metric in self.spec.metrics:
            if metric == "mae":
                results["mae"] = float(np.mean(np.abs(y_true - y_pred)))

            elif metric == "rmse":
                results["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            elif metric == "mape":
                denom = np.where(y_true == 0, np.nan, y_true)
                ratio = np.abs((y_true - y_pred) / denom) * 100
                results["mape"] = _nanmean_or_nan(ratio)

            elif metric == "smape":
                denom = np.abs(y_true) + np.abs(y_pred)
                ratio = np.where(denom == 0, np.nan, 2.0 * np.abs(y_true - y_pred) / denom) * 100
                results["smape"] = _nanmean_or_nan(ratio)

            elif metric == "wape":
                denom = np.sum(np.abs(y_true))
                results["wape"] = float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom != 0 else float("nan")

        return results

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Eden is not fitted yet.")


__all__ = [
    "AdapterStrategy",
    "BacktestConfig",
    "ConformalIntervalAdapter",
    "Eden",
    "EdenBahamutBundle",
    "EdenSpec",
    "EdenSplits",
    "ForecastAdapter",
    "ProphetPerTargetAdapter",
    "QuantileEnsembleAdapter",
    "SeriesStructure",
    "SklearnMultiOutputAdapter",
    "SklearnPerTargetAdapter",
    "SplitName",
    "SUPPORTED_METRICS",
]