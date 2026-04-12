from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .core import (
    Eden,
    EdenBahamutBundle,
    EdenSpec,
    EdenSplits,
    ForecastAdapter,
    ProphetPerTargetAdapter,
    QuantileEnsembleAdapter,
)

try:
    import ipywidgets as widgets
    from IPython import get_ipython
    from IPython.display import clear_output, display
except Exception:
    widgets = None
    clear_output = None
    display = None
    get_ipython = None


def _eden_require_widgets() -> None:
    if widgets is None or clear_output is None or display is None:
        raise ImportError("ipywidgets e IPython son necesarios para usar la tabla interactiva de Eden.")


def _eden_resolve_namespace(namespace: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if namespace is not None:
        return dict(namespace)

    if get_ipython is None:
        return {}

    shell = get_ipython()
    if shell is None or not hasattr(shell, "user_ns"):
        return {}
    return dict(shell.user_ns)


def _eden_publish_notebook_bindings(**values: Any) -> None:
    if get_ipython is None:
        return

    shell = get_ipython()
    if shell is None or not hasattr(shell, "user_ns"):
        return

    shell.user_ns.update(values)


def _eden_is_bahamut_segments(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return "indices_train" in value and any(key.startswith("X_") for key in value)


def _eden_is_indexer_candidate(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping, pd.DataFrame, pd.Series)):
        return False
    if isinstance(value, pd.Index):
        return True
    if isinstance(value, Sequence):
        return len(value) > 0
    return False


def _eden_is_model_blueprint_candidate(value: Any) -> bool:
    if value is None or isinstance(value, (Eden, ForecastAdapter, pd.DataFrame, pd.Series, Mapping)):
        return False
    if isinstance(value, type):
        return value is not Eden and not issubclass(value, ForecastAdapter)
    return hasattr(value, "fit") and hasattr(value, "predict")


def _eden_is_adapter_blueprint_candidate(value: Any) -> bool:
    return isinstance(value, ForecastAdapter)


def _eden_require_sklearn_linear_models() -> tuple[Any, Any]:
    try:
        from sklearn.linear_model import LinearRegression, QuantileRegressor
    except Exception as exc:
        raise ImportError(
            "Los presets lineales de Eden requieren scikit-learn. Instala './eden[notebook]' o './eden[dev]'."
        ) from exc
    return LinearRegression, QuantileRegressor


def _eden_require_prophet_runtime() -> None:
    try:
        import prophet  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "El preset Prophet de Eden requiere instalar './eden[prophet]' o './eden[notebook,prophet]'."
        ) from exc


def _eden_interval_bounds(coverage: float) -> tuple[float, float]:
    normalized = float(coverage)
    if not 0 < normalized < 1:
        raise ValueError("interval_coverage debe estar entre 0 y 1.")
    tail = (1.0 - normalized) / 2.0
    return tail, 1.0 - tail


def _eden_read_dataframe_from_buffer(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(content)

    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(buffer, sep=None, engine="python")
        except Exception:
            buffer.seek(0)
            return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)
    if suffix == ".json":
        return pd.read_json(buffer)
    if suffix == ".parquet":
        return pd.read_parquet(buffer)

    raise ValueError(f"Formato de dataframe no soportado: {suffix or filename}")


def _eden_read_dataframe_from_path(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(resolved, sep=None, engine="python")
        except Exception:
            return pd.read_csv(resolved)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(resolved)
    if suffix == ".json":
        return pd.read_json(resolved)
    if suffix == ".parquet":
        return pd.read_parquet(resolved)

    raise ValueError(f"Formato de dataframe no soportado: {resolved.suffix or resolved.name}")


def _eden_read_indexer_from_buffer(content: bytes, filename: str) -> list[Any]:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(content)

    if suffix == ".json":
        payload = json.loads(content.decode("utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            for key in ("indices", "index", "rows"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise ValueError("El JSON de indices debe ser una lista o un objeto con claves 'indices', 'index' o 'rows'.")

    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(buffer, header=None)
        return frame.iloc[:, 0].tolist()

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(buffer, header=None)
        return frame.iloc[:, 0].tolist()

    raise ValueError(f"Formato de indices no soportado: {suffix or filename}")


def _eden_read_indexer_from_path(path: str | Path) -> list[Any]:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            for key in ("indices", "index", "rows"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise ValueError("El JSON de indices debe ser una lista o un objeto con claves 'indices', 'index' o 'rows'.")

    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(resolved, header=None)
        return frame.iloc[:, 0].tolist()

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(resolved, header=None)
        return frame.iloc[:, 0].tolist()

    raise ValueError(f"Formato de indices no soportado: {resolved.suffix or resolved.name}")


def _eden_first_upload(value: Any) -> tuple[str, bytes] | None:
    if not value:
        return None

    if isinstance(value, dict):
        item = next(iter(value.values()))
        if isinstance(item, Mapping):
            return str(item.get("name") or next(iter(value.keys()))), bytes(item.get("content", b""))

    if isinstance(value, (list, tuple)):
        item = value[0]
        if isinstance(item, Mapping):
            return str(item.get("name") or "upload"), bytes(item.get("content", b""))

    return None


def _eden_build_splits_from_column(
    df: pd.DataFrame,
    split_col: str,
    train_value: str = "train",
    valid_value: str = "valid",
    test_value: str = "test",
) -> EdenSplits:
    normalized = df[split_col].astype(str).str.strip().str.lower()
    train_idx = df.index[normalized == str(train_value).strip().lower()].tolist()
    valid_idx = df.index[normalized == str(valid_value).strip().lower()].tolist()
    test_idx = df.index[normalized == str(test_value).strip().lower()].tolist()

    return EdenSplits(
        train_idx=train_idx,
        valid_idx=valid_idx or None,
        test_idx=test_idx or None,
    )


class _PreviewEstimator:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "_PreviewEstimator":
        return self

    def predict(self, x: pd.DataFrame) -> list[float]:
        return [0.0] * len(x)


@dataclass(slots=True)
class EdenWorkbenchResult:
    source_mode: str
    source_name: str | None
    df: pd.DataFrame
    splits: EdenSplits
    spec: EdenSpec
    partitions: dict[str, pd.DataFrame]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for split_name, frame in self.partitions.items():
            row: dict[str, Any] = {"split": split_name, "rows": len(frame)}
            if not frame.empty and self.spec.timestamp_col in frame.columns:
                row["start"] = frame[self.spec.timestamp_col].min()
                row["end"] = frame[self.spec.timestamp_col].max()
            rows.append(row)
        return pd.DataFrame(rows)

    def preview(self, split_name: str = "train", rows: int = 6) -> pd.DataFrame:
        if split_name not in self.partitions:
            raise ValueError(f"Split '{split_name}' no esta disponible. Disponibles: {sorted(self.partitions)}")
        return self.partitions[split_name].head(int(rows)).copy()


@dataclass(slots=True)
class EdenWorkbenchRunResult:
    source_result: EdenWorkbenchResult
    runtime_kind: str
    runtime_name: str
    prediction_split: str
    eden_model: Eden
    selected_predictions: pd.DataFrame
    selected_metrics: dict[str, float] | None
    performance_report: pd.DataFrame
    diagnostics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "runtime_kind": self.runtime_kind,
                    "runtime_name": self.runtime_name,
                    "prediction_split": self.prediction_split,
                    "prediction_rows": len(self.selected_predictions),
                    "supports_interval": self.diagnostics.get("supports_interval"),
                    "interval_calibration_source": self.diagnostics.get("interval_calibration_source"),
                }
            ]
        )

    def metrics_frame(self) -> pd.DataFrame:
        if self.selected_metrics is None:
            return pd.DataFrame({"info": ["No hay métricas disponibles para el split seleccionado."]})
        return pd.Series(self.selected_metrics, name="valor").to_frame()

    def diagnostics_frame(self) -> pd.DataFrame:
        return pd.Series(self.diagnostics, name="valor").to_frame()


class EdenWorkbench:
    def __init__(self, namespace: Mapping[str, Any] | None = None):
        self.namespace = _eden_resolve_namespace(namespace)
        self.last_result_: EdenWorkbenchResult | None = None
        self.last_run_: EdenWorkbenchRunResult | None = None

    def refresh_namespace(self, namespace: Mapping[str, Any] | None = None) -> None:
        self.namespace = _eden_resolve_namespace(namespace)

    def discover_sources(self) -> dict[str, list[str]]:
        dataframes = sorted(
            name for name, value in self.namespace.items() if isinstance(value, pd.DataFrame)
        )
        bahamut_segments = sorted(
            name for name, value in self.namespace.items() if _eden_is_bahamut_segments(value)
        )
        indexers = sorted(
            name for name, value in self.namespace.items() if _eden_is_indexer_candidate(value)
        )
        model_blueprints = sorted(
            name for name, value in self.namespace.items() if _eden_is_model_blueprint_candidate(value)
        )
        adapter_blueprints = sorted(
            name for name, value in self.namespace.items() if _eden_is_adapter_blueprint_candidate(value)
        )
        return {
            "dataframes": dataframes,
            "bahamut_segments": bahamut_segments,
            "indexers": indexers,
            "model_blueprints": model_blueprints,
            "adapter_blueprints": adapter_blueprints,
        }

    def materialize_from_bahamut(
        self,
        bundle_name: str,
        *,
        timestamp_col: str,
        target_cols: Sequence[str],
        feature_cols: Sequence[str] | None = None,
        entity_id_col: str | None = None,
        df_name: str | None = None,
        spec_kwargs: Mapping[str, Any] | None = None,
    ) -> EdenWorkbenchResult:
        if bundle_name not in self.namespace:
            raise KeyError(f"No existe la variable '{bundle_name}' en el namespace del notebook.")

        source = self.namespace[bundle_name]
        if not _eden_is_bahamut_segments(source):
            raise TypeError(f"La variable '{bundle_name}' no tiene pinta de ser un bundle de Bahamut.")

        original_df = None
        if df_name is not None:
            original_df = self._require_dataframe(df_name)

        bundle = EdenBahamutBundle.from_bahamut(
            source,
            df=original_df,
            timestamp_col=timestamp_col,
            entity_id_col=entity_id_col,
            feature_cols=feature_cols,
            target_cols=target_cols,
        )
        resolved_features = list(feature_cols) if feature_cols is not None else bundle.feature_cols

        result = self._build_result(
            df=bundle.df,
            splits=bundle.splits,
            timestamp_col=timestamp_col,
            target_cols=target_cols,
            feature_cols=resolved_features,
            entity_id_col=entity_id_col,
            source_mode="bahamut",
            source_name=bundle_name,
            metadata={
                "bahamut_feature_cols": bundle.feature_cols,
                "bahamut_target_cols": bundle.target_cols,
                "bahamut_metadata": bundle.metadata,
            },
            spec_kwargs=spec_kwargs,
        )
        self.last_result_ = result
        return result

    def materialize_from_variables(
        self,
        df_name: str,
        *,
        timestamp_col: str,
        target_cols: Sequence[str],
        feature_cols: Sequence[str] | None = None,
        entity_id_col: str | None = None,
        train_idx_name: str | None = None,
        valid_idx_name: str | None = None,
        test_idx_name: str | None = None,
        split_col: str | None = None,
        train_value: str = "train",
        valid_value: str = "valid",
        test_value: str = "test",
        spec_kwargs: Mapping[str, Any] | None = None,
    ) -> EdenWorkbenchResult:
        df = self._require_dataframe(df_name)
        if split_col is not None:
            splits = _eden_build_splits_from_column(
                df,
                split_col=split_col,
                train_value=train_value,
                valid_value=valid_value,
                test_value=test_value,
            )
        else:
            splits = EdenSplits(
                train_idx=self._require_indexer(train_idx_name, role="train"),
                valid_idx=self._require_optional_indexer(valid_idx_name),
                test_idx=self._require_optional_indexer(test_idx_name),
            )

        result = self._build_result(
            df=df,
            splits=splits,
            timestamp_col=timestamp_col,
            target_cols=target_cols,
            feature_cols=feature_cols,
            entity_id_col=entity_id_col,
            source_mode="variables",
            source_name=df_name,
            metadata={
                "split_col": split_col,
                "train_idx_name": train_idx_name,
                "valid_idx_name": valid_idx_name,
                "test_idx_name": test_idx_name,
            },
            spec_kwargs=spec_kwargs,
        )
        self.last_result_ = result
        return result

    def materialize_from_files(
        self,
        df_source: str | Path,
        *,
        timestamp_col: str,
        target_cols: Sequence[str],
        feature_cols: Sequence[str] | None = None,
        entity_id_col: str | None = None,
        train_idx_source: str | Path | None = None,
        valid_idx_source: str | Path | None = None,
        test_idx_source: str | Path | None = None,
        split_col: str | None = None,
        train_value: str = "train",
        valid_value: str = "valid",
        test_value: str = "test",
        spec_kwargs: Mapping[str, Any] | None = None,
    ) -> EdenWorkbenchResult:
        df = _eden_read_dataframe_from_path(df_source)
        if split_col is not None:
            splits = _eden_build_splits_from_column(
                df,
                split_col=split_col,
                train_value=train_value,
                valid_value=valid_value,
                test_value=test_value,
            )
        else:
            splits = EdenSplits(
                train_idx=_eden_read_indexer_from_path(train_idx_source) if train_idx_source is not None else None,
                valid_idx=_eden_read_indexer_from_path(valid_idx_source) if valid_idx_source is not None else None,
                test_idx=_eden_read_indexer_from_path(test_idx_source) if test_idx_source is not None else None,
            )

        result = self._build_result(
            df=df,
            splits=splits,
            timestamp_col=timestamp_col,
            target_cols=target_cols,
            feature_cols=feature_cols,
            entity_id_col=entity_id_col,
            source_mode="files",
            source_name=str(df_source),
            metadata={
                "train_idx_source": None if train_idx_source is None else str(train_idx_source),
                "valid_idx_source": None if valid_idx_source is None else str(valid_idx_source),
                "test_idx_source": None if test_idx_source is None else str(test_idx_source),
                "split_col": split_col,
            },
            spec_kwargs=spec_kwargs,
        )
        self.last_result_ = result
        return result

    def build_eden(self, model: Any | None = None, adapter: Any | None = None) -> Eden:
        if self.last_result_ is None:
            raise RuntimeError("Todavia no hay una configuracion materializada en la tabla interactiva.")
        return Eden(spec=self.last_result_.spec, model=model, adapter=adapter)

    def train_and_predict(
        self,
        *,
        runtime: str = "linear",
        prediction_split: str = "test",
        namespace_model_name: str | None = None,
        namespace_adapter_name: str | None = None,
        include_train_metrics: bool = False,
        interval_coverage: float | None = None,
    ) -> EdenWorkbenchRunResult:
        if self.last_result_ is None:
            raise RuntimeError("Todavia no hay una configuracion materializada en la tabla interactiva.")

        runtime_spec = deepcopy(self.last_result_.spec)
        if interval_coverage is not None:
            runtime_spec.interval_coverage = float(interval_coverage)

        model, adapter, runtime_name = self._resolve_runtime(
            runtime,
            spec=runtime_spec,
            namespace_model_name=namespace_model_name,
            namespace_adapter_name=namespace_adapter_name,
        )
        eden_model = Eden(spec=runtime_spec, model=model, adapter=adapter)
        eden_model.fit(self.last_result_.df, self.last_result_.splits)

        if prediction_split not in self.last_result_.partitions:
            raise ValueError(
                f"Split '{prediction_split}' no esta disponible. Disponibles: {sorted(self.last_result_.partitions)}"
            )

        selected_bundle = eden_model.predict_with_metrics(self.last_result_.partitions[prediction_split])
        performance_report = eden_model.performance_report(
            self.last_result_.df,
            self.last_result_.splits,
            include_train=include_train_metrics,
        )

        run_result = EdenWorkbenchRunResult(
            source_result=self.last_result_,
            runtime_kind=runtime,
            runtime_name=runtime_name,
            prediction_split=prediction_split,
            eden_model=eden_model,
            selected_predictions=selected_bundle["predictions"],
            selected_metrics=selected_bundle["metrics"],
            performance_report=performance_report,
            diagnostics=eden_model.diagnostics(),
            metadata={
                "interval_coverage": runtime_spec.interval_coverage,
                "include_train_metrics": include_train_metrics,
            },
        )
        self.last_run_ = run_result
        return run_result

    def _resolve_runtime(
        self,
        runtime: str,
        *,
        spec: EdenSpec,
        namespace_model_name: str | None,
        namespace_adapter_name: str | None,
    ) -> tuple[Any | None, ForecastAdapter | None, str]:
        if runtime == "linear":
            LinearRegression, _ = _eden_require_sklearn_linear_models()
            return LinearRegression(), None, "LinearRegression"

        if runtime == "quantile_linear":
            LinearRegression, QuantileRegressor = _eden_require_sklearn_linear_models()
            coverage = 0.9 if spec.interval_coverage is None else float(spec.interval_coverage)
            lower_q, upper_q = _eden_interval_bounds(coverage)
            adapter = QuantileEnsembleAdapter(
                center_estimator_blueprint=LinearRegression(),
                lower_estimator_blueprint=QuantileRegressor(quantile=lower_q, alpha=0.0),
                upper_estimator_blueprint=QuantileRegressor(quantile=upper_q, alpha=0.0),
                lower_quantile=lower_q,
                upper_quantile=upper_q,
            )
            return None, adapter, f"QuantileRegressor[{int(round(coverage * 100))}]"

        if runtime == "prophet":
            _eden_require_prophet_runtime()
            coverage = 0.8 if spec.interval_coverage is None else float(spec.interval_coverage)
            adapter = ProphetPerTargetAdapter(
                timestamp_col=spec.timestamp_col,
                entity_id_col=spec.entity_id_col,
                interval_width=coverage,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            return None, adapter, f"Prophet[{int(round(coverage * 100))}]"

        if runtime == "namespace_model":
            if namespace_model_name is None:
                raise ValueError("Selecciona un modelo del notebook para ejecutar Eden.")
            if namespace_model_name not in self.namespace:
                raise KeyError(f"No existe la variable '{namespace_model_name}' en el namespace del notebook.")
            model_blueprint = self.namespace[namespace_model_name]
            if not _eden_is_model_blueprint_candidate(model_blueprint):
                raise TypeError(f"La variable '{namespace_model_name}' no es un blueprint de modelo valido.")
            return model_blueprint, None, namespace_model_name

        if runtime == "namespace_adapter":
            if namespace_adapter_name is None:
                raise ValueError("Selecciona un adapter del notebook para ejecutar Eden.")
            if namespace_adapter_name not in self.namespace:
                raise KeyError(f"No existe la variable '{namespace_adapter_name}' en el namespace del notebook.")
            adapter_blueprint = self.namespace[namespace_adapter_name]
            if not _eden_is_adapter_blueprint_candidate(adapter_blueprint):
                raise TypeError(f"La variable '{namespace_adapter_name}' no es un adapter valido.")
            return None, adapter_blueprint, namespace_adapter_name

        raise ValueError(
            f"Runtime '{runtime}' no soportado. Usa uno de: linear, quantile_linear, prophet, namespace_model, namespace_adapter."
        )

    def tabla_interactiva(self, preview_rows: int = 6):
        _eden_require_widgets()
        self.refresh_namespace(self.namespace)

        sources = self.discover_sources()
        state: dict[str, Any] = {"result": None, "run_result": None, "config_frame": None}

        source_widget = widgets.Dropdown(
            options=[
                ("Bahamut", "bahamut"),
                ("Variables del notebook", "variables"),
                ("Archivos locales", "files"),
            ],
            value="bahamut" if sources["bahamut_segments"] else "variables",
            description="Fuente",
            layout=widgets.Layout(width="300px"),
        )
        bundle_widget = widgets.Dropdown(
            options=[("Selecciona bundle", "__none__")] + [(name, name) for name in sources["bahamut_segments"]],
            value="__none__",
            description="Bahamut",
            layout=widgets.Layout(width="320px"),
        )
        original_df_widget = widgets.Dropdown(
            options=[("Sin df original", "__none__")] + [(name, name) for name in sources["dataframes"]],
            value="__none__",
            description="df original",
            layout=widgets.Layout(width="320px"),
        )
        df_widget = widgets.Dropdown(
            options=[("Selecciona df", "__none__")] + [(name, name) for name in sources["dataframes"]],
            value="__none__",
            description="DataFrame",
            layout=widgets.Layout(width="320px"),
        )
        variable_split_mode_widget = widgets.Dropdown(
            options=[
                ("Variables de indices", "indices"),
                ("Columna de split", "split_col"),
            ],
            value="indices",
            description="Split",
            layout=widgets.Layout(width="280px"),
        )
        train_idx_widget = widgets.Dropdown(
            options=[("Selecciona train_idx", "__none__")] + [(name, name) for name in sources["indexers"]],
            value="__none__",
            description="train_idx",
            layout=widgets.Layout(width="300px"),
        )
        valid_idx_widget = widgets.Dropdown(
            options=[("Sin valid_idx", "__none__")] + [(name, name) for name in sources["indexers"]],
            value="__none__",
            description="valid_idx",
            layout=widgets.Layout(width="300px"),
        )
        test_idx_widget = widgets.Dropdown(
            options=[("Sin test_idx", "__none__")] + [(name, name) for name in sources["indexers"]],
            value="__none__",
            description="test_idx",
            layout=widgets.Layout(width="300px"),
        )
        file_df_path_widget = widgets.Text(
            value="",
            description="Archivo df",
            placeholder=r"C:\ruta\dataset.csv o .xlsx",
            layout=widgets.Layout(width="420px"),
        )
        file_df_upload_widget = widgets.FileUpload(
            accept=".csv,.txt,.xlsx,.xls,.json,.parquet",
            multiple=False,
            description="Subir df",
        )
        file_split_mode_widget = widgets.Dropdown(
            options=[
                ("Archivos de indices", "indices"),
                ("Columna de split", "split_col"),
            ],
            value="indices",
            description="Split",
            layout=widgets.Layout(width="280px"),
        )
        file_train_idx_path_widget = widgets.Text(
            value="",
            description="Train file",
            placeholder=r"C:\ruta\train_idx.csv o .json",
            layout=widgets.Layout(width="420px"),
        )
        file_valid_idx_path_widget = widgets.Text(
            value="",
            description="Valid file",
            placeholder=r"C:\ruta\valid_idx.csv o .json",
            layout=widgets.Layout(width="420px"),
        )
        file_test_idx_path_widget = widgets.Text(
            value="",
            description="Test file",
            placeholder=r"C:\ruta\test_idx.csv o .json",
            layout=widgets.Layout(width="420px"),
        )
        file_train_upload_widget = widgets.FileUpload(
            accept=".csv,.txt,.xlsx,.xls,.json",
            multiple=False,
            description="Subir train",
        )
        file_valid_upload_widget = widgets.FileUpload(
            accept=".csv,.txt,.xlsx,.xls,.json",
            multiple=False,
            description="Subir valid",
        )
        file_test_upload_widget = widgets.FileUpload(
            accept=".csv,.txt,.xlsx,.xls,.json",
            multiple=False,
            description="Subir test",
        )
        split_col_widget = widgets.Dropdown(
            options=[("Sin columna split", "__none__")],
            value="__none__",
            description="Col split",
            layout=widgets.Layout(width="300px"),
        )
        train_label_widget = widgets.Text(value="train", description="Valor train", layout=widgets.Layout(width="240px"))
        valid_label_widget = widgets.Text(value="valid", description="Valor valid", layout=widgets.Layout(width="240px"))
        test_label_widget = widgets.Text(value="test", description="Valor test", layout=widgets.Layout(width="240px"))
        timestamp_widget = widgets.Dropdown(
            options=[("Selecciona timestamp", "__none__")],
            value="__none__",
            description="Timestamp",
            layout=widgets.Layout(width="300px"),
        )
        entity_widget = widgets.Dropdown(
            options=[("Sin entidad", "__none__")],
            value="__none__",
            description="Entidad",
            layout=widgets.Layout(width="300px"),
        )
        target_widget = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Target",
            rows=6,
            layout=widgets.Layout(width="280px", height="180px"),
        )
        feature_widget = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Features",
            rows=10,
            layout=widgets.Layout(width="320px", height="220px"),
        )
        runtime_widget = widgets.Dropdown(
            options=[
                ("Lineal sklearn", "linear"),
                ("Cuantiles lineales", "quantile_linear"),
                ("Prophet", "prophet"),
                ("Modelo notebook", "namespace_model"),
                ("Adapter notebook", "namespace_adapter"),
            ],
            value="linear",
            description="Modelo",
            layout=widgets.Layout(width="300px"),
        )
        namespace_model_widget = widgets.Dropdown(
            options=[("Selecciona modelo", "__none__")] + [(name, name) for name in sources["model_blueprints"]],
            value="__none__",
            description="Modelo ns",
            layout=widgets.Layout(width="320px"),
        )
        namespace_adapter_widget = widgets.Dropdown(
            options=[("Selecciona adapter", "__none__")] + [(name, name) for name in sources["adapter_blueprints"]],
            value="__none__",
            description="Adapter ns",
            layout=widgets.Layout(width="320px"),
        )
        prediction_split_widget = widgets.Dropdown(
            options=[("Train", "train"), ("Valid", "valid"), ("Test", "test")],
            value="test",
            description="Predecir",
            layout=widgets.Layout(width="220px"),
        )
        interval_coverage_widget = widgets.FloatSlider(
            value=0.9,
            min=0.5,
            max=0.99,
            step=0.05,
            readout_format=".2f",
            description="Cobertura",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
        )
        include_train_metrics_widget = widgets.Checkbox(value=False, description="Reporte con train")
        view_widget = widgets.Dropdown(
            options=[
                ("Resumen", "summary"),
                ("Preview", "preview"),
                ("Config", "config"),
                ("Predicciones", "predictions"),
                ("Reporte", "report"),
                ("Diagnóstico", "diagnostics"),
            ],
            value="summary",
            description="Vista",
            layout=widgets.Layout(width="260px"),
        )
        preview_split_widget = widgets.Dropdown(
            options=[("Train", "train"), ("Valid", "valid"), ("Test", "test")],
            value="train",
            description="Preview",
            layout=widgets.Layout(width="220px"),
        )
        preview_rows_widget = widgets.IntSlider(
            value=max(3, int(preview_rows)),
            min=3,
            max=20,
            step=1,
            description="Filas",
            continuous_update=False,
            layout=widgets.Layout(width="260px"),
        )
        refresh_button = widgets.Button(
            description="Preparar Eden",
            button_style="primary",
            icon="refresh",
        )
        train_button = widgets.Button(
            description="Entrenar y predecir",
            button_style="success",
            icon="play",
        )

        status_output = widgets.Output()
        summary_output = widgets.Output()
        table_output = widgets.Output()

        bahamut_box = widgets.VBox([bundle_widget, original_df_widget])
        variable_indices_box = widgets.VBox([train_idx_widget, valid_idx_widget, test_idx_widget])
        variable_split_col_box = widgets.VBox([split_col_widget, widgets.HBox([train_label_widget, valid_label_widget, test_label_widget])])
        variables_box = widgets.VBox([df_widget, variable_split_mode_widget, variable_indices_box, variable_split_col_box])
        file_indices_box = widgets.VBox([
            widgets.HBox([file_train_idx_path_widget, file_train_upload_widget]),
            widgets.HBox([file_valid_idx_path_widget, file_valid_upload_widget]),
            widgets.HBox([file_test_idx_path_widget, file_test_upload_widget]),
        ])
        files_box = widgets.VBox([
            widgets.HBox([file_df_path_widget, file_df_upload_widget]),
            file_split_mode_widget,
            file_indices_box,
            widgets.VBox([split_col_widget, widgets.HBox([train_label_widget, valid_label_widget, test_label_widget])]),
        ])
        namespace_model_box = widgets.VBox([namespace_model_widget])
        namespace_adapter_box = widgets.VBox([namespace_adapter_widget])

        def _uploaded_dataframe() -> pd.DataFrame | None:
            uploaded = _eden_first_upload(file_df_upload_widget.value)
            if uploaded is None:
                return None
            filename, content = uploaded
            return _eden_read_dataframe_from_buffer(content, filename)

        def _resolve_current_df() -> pd.DataFrame | None:
            if source_widget.value == "bahamut":
                if bundle_widget.value == "__none__":
                    return None
                source = self.namespace.get(bundle_widget.value)
                if not _eden_is_bahamut_segments(source):
                    return None
                original_df = None if original_df_widget.value == "__none__" else self.namespace.get(original_df_widget.value)
                try:
                    bundle = EdenBahamutBundle.from_bahamut(
                        source,
                        df=original_df,
                        timestamp_col=None,
                        entity_id_col=None,
                    )
                    return bundle.df
                except Exception:
                    return original_df if isinstance(original_df, pd.DataFrame) else None

            if source_widget.value == "variables":
                if df_widget.value == "__none__":
                    return None
                value = self.namespace.get(df_widget.value)
                return value.copy() if isinstance(value, pd.DataFrame) else None

            if file_df_path_widget.value.strip():
                try:
                    return _eden_read_dataframe_from_path(file_df_path_widget.value.strip())
                except Exception:
                    pass
            return _uploaded_dataframe()

        def _split_col_options(columns: list[str]) -> list[tuple[str, str]]:
            return [("Sin columna split", "__none__")] + [(column, column) for column in columns]

        def _default_targets(columns: list[str]) -> tuple[str, ...]:
            preferred = [column for column in columns if column.lower().startswith(("y", "target"))]
            return tuple(preferred[:1])

        def _build_config_frame(result: EdenWorkbenchResult) -> pd.DataFrame:
            return pd.Series(
                {
                    "source_mode": result.source_mode,
                    "source_name": result.source_name,
                    "timestamp_col": result.spec.timestamp_col,
                    "entity_id_col": result.spec.entity_id_col,
                    "series_structure": result.spec.series_structure,
                    "target_cols": result.spec.target_cols,
                    "feature_cols": result.spec.feature_cols,
                    **result.metadata,
                },
                name="valor",
            ).to_frame()

        def _info_frame(message: str) -> pd.DataFrame:
            return pd.DataFrame({"info": [message]})

        def _sync_split_widgets() -> None:
            if state["result"] is None:
                options = [("Train", "train"), ("Valid", "valid"), ("Test", "test")]
            else:
                options = [(name.capitalize(), name) for name in state["result"].partitions]

            preview_split_widget.options = options
            if preview_split_widget.value not in dict(options).values():
                preview_split_widget.value = options[0][1]

            prediction_split_widget.options = options
            preferred_prediction = "test" if "test" in dict(options).values() else options[-1][1]
            if prediction_split_widget.value not in dict(options).values():
                prediction_split_widget.value = preferred_prediction

        def _render_views() -> None:
            result = state["result"]
            run_result = state["run_result"]
            config_frame = state["config_frame"]

            with summary_output:
                clear_output(wait=True)
                if result is None or config_frame is None:
                    return
                print("Configuracion resuelta")
                display(config_frame)
                if run_result is not None:
                    print("Ejecucion del modelo")
                    display(run_result.summary())
                    print("Metricas del split seleccionado")
                    display(run_result.metrics_frame())

            with table_output:
                clear_output(wait=True)
                if result is None or config_frame is None:
                    return

                if view_widget.value == "config":
                    display(config_frame)
                elif view_widget.value == "preview":
                    display(result.preview(preview_split_widget.value, rows=int(preview_rows_widget.value)))
                elif view_widget.value == "predictions":
                    if run_result is None:
                        display(_info_frame("Entrena un modelo desde la tabla para ver predicciones."))
                    else:
                        display(run_result.selected_predictions)
                elif view_widget.value == "report":
                    if run_result is None:
                        display(_info_frame("Entrena un modelo para generar el reporte de desempeño."))
                    else:
                        display(run_result.performance_report)
                elif view_widget.value == "diagnostics":
                    if run_result is None:
                        display(_info_frame("Entrena un modelo para ver el diagnostico de Eden."))
                    else:
                        display(run_result.diagnostics_frame())
                else:
                    display(result.summary())

        def _sync_runtime_controls(*_args) -> None:
            current_sources = self.discover_sources()
            model_options = [("Selecciona modelo", "__none__")] + [
                (name, name) for name in current_sources["model_blueprints"]
            ]
            adapter_options = [("Selecciona adapter", "__none__")] + [
                (name, name) for name in current_sources["adapter_blueprints"]
            ]
            namespace_model_widget.options = model_options
            namespace_adapter_widget.options = adapter_options
            if namespace_model_widget.value not in dict(model_options).values():
                namespace_model_widget.value = "__none__"
            if namespace_adapter_widget.value not in dict(adapter_options).values():
                namespace_adapter_widget.value = "__none__"

            namespace_model_box.layout.display = "flex" if runtime_widget.value == "namespace_model" else "none"
            namespace_adapter_box.layout.display = "flex" if runtime_widget.value == "namespace_adapter" else "none"

        def _sync_columns(*_args) -> None:
            current_df = _resolve_current_df()
            columns = [] if current_df is None else current_df.columns.tolist()

            current_timestamp = timestamp_widget.value if timestamp_widget.value in columns else "__none__"
            timestamp_widget.options = [("Selecciona timestamp", "__none__")] + [(column, column) for column in columns]
            timestamp_widget.value = current_timestamp

            entity_options = [("Sin entidad", "__none__")] + [
                (column, column) for column in columns if column != timestamp_widget.value
            ]
            current_entity = entity_widget.value if entity_widget.value in {value for _, value in entity_options} else "__none__"
            entity_widget.options = entity_options
            entity_widget.value = current_entity

            split_col_widget.options = _split_col_options(columns)
            if split_col_widget.value not in {value for _, value in split_col_widget.options}:
                split_col_widget.value = "__none__"

            target_options = columns
            current_targets = tuple(value for value in target_widget.value if value in target_options)
            if not current_targets and columns:
                current_targets = _default_targets(columns)
            target_widget.options = target_options
            target_widget.value = current_targets

            excluded = set(target_widget.value)
            if timestamp_widget.value != "__none__":
                excluded.add(timestamp_widget.value)
            if entity_widget.value != "__none__":
                excluded.add(entity_widget.value)
            if split_col_widget.value != "__none__":
                excluded.add(split_col_widget.value)

            feature_options = [column for column in columns if column not in excluded]
            current_features = tuple(value for value in feature_widget.value if value in feature_options)
            if not current_features and feature_options:
                current_features = tuple(feature_options)
            feature_widget.options = feature_options
            feature_widget.value = current_features

            bahamut_box.layout.display = "flex" if source_widget.value == "bahamut" else "none"
            variables_box.layout.display = "flex" if source_widget.value == "variables" else "none"
            files_box.layout.display = "flex" if source_widget.value == "files" else "none"

            variable_indices_box.layout.display = "flex" if variable_split_mode_widget.value == "indices" else "none"
            variable_split_col_box.layout.display = "flex" if variable_split_mode_widget.value == "split_col" and source_widget.value == "variables" else "none"
            file_indices_box.layout.display = "flex" if file_split_mode_widget.value == "indices" else "none"
            _sync_runtime_controls()

        def _build_spec_kwargs() -> dict[str, Any]:
            return {
                "metrics": ["mae", "rmse", "wape"],
                "prediction_suffix": "_pred",
            }

        def _build_from_widgets() -> EdenWorkbenchResult:
            if timestamp_widget.value == "__none__":
                raise ValueError("Selecciona una columna temporal para Eden.")
            if not target_widget.value:
                raise ValueError("Selecciona al menos un target.")

            feature_cols = list(feature_widget.value)
            if not feature_cols:
                raise ValueError("Selecciona al menos una feature.")

            entity_id_col = None if entity_widget.value == "__none__" else entity_widget.value
            spec_kwargs = _build_spec_kwargs()

            if source_widget.value == "bahamut":
                if bundle_widget.value == "__none__":
                    raise ValueError("Selecciona un bundle de Bahamut.")
                return self.materialize_from_bahamut(
                    bundle_widget.value,
                    timestamp_col=timestamp_widget.value,
                    target_cols=list(target_widget.value),
                    feature_cols=feature_cols,
                    entity_id_col=entity_id_col,
                    df_name=None if original_df_widget.value == "__none__" else original_df_widget.value,
                    spec_kwargs=spec_kwargs,
                )

            if source_widget.value == "variables":
                if df_widget.value == "__none__":
                    raise ValueError("Selecciona un DataFrame del notebook.")
                if variable_split_mode_widget.value == "split_col":
                    if split_col_widget.value == "__none__":
                        raise ValueError("Selecciona una columna que identifique train/valid/test.")
                    return self.materialize_from_variables(
                        df_widget.value,
                        timestamp_col=timestamp_widget.value,
                        target_cols=list(target_widget.value),
                        feature_cols=feature_cols,
                        entity_id_col=entity_id_col,
                        split_col=split_col_widget.value,
                        train_value=train_label_widget.value,
                        valid_value=valid_label_widget.value,
                        test_value=test_label_widget.value,
                        spec_kwargs=spec_kwargs,
                    )

                return self.materialize_from_variables(
                    df_widget.value,
                    timestamp_col=timestamp_widget.value,
                    target_cols=list(target_widget.value),
                    feature_cols=feature_cols,
                    entity_id_col=entity_id_col,
                    train_idx_name=None if train_idx_widget.value == "__none__" else train_idx_widget.value,
                    valid_idx_name=None if valid_idx_widget.value == "__none__" else valid_idx_widget.value,
                    test_idx_name=None if test_idx_widget.value == "__none__" else test_idx_widget.value,
                    spec_kwargs=spec_kwargs,
                )

            uploaded_df = _eden_first_upload(file_df_upload_widget.value)
            if file_df_path_widget.value.strip():
                df_source: str | Path = file_df_path_widget.value.strip()
            elif uploaded_df is not None:
                filename, content = uploaded_df
                df = _eden_read_dataframe_from_buffer(content, filename)
                return self._build_result(
                    df=df,
                    splits=self._build_file_splits(df, file_split_mode_widget.value, split_col_widget.value, train_label_widget.value, valid_label_widget.value, test_label_widget.value, file_train_idx_path_widget.value, file_valid_idx_path_widget.value, file_test_idx_path_widget.value, file_train_upload_widget.value, file_valid_upload_widget.value, file_test_upload_widget.value),
                    timestamp_col=timestamp_widget.value,
                    target_cols=list(target_widget.value),
                    feature_cols=feature_cols,
                    entity_id_col=entity_id_col,
                    source_mode="files",
                    source_name=filename,
                    metadata={"uploaded": True},
                    spec_kwargs=spec_kwargs,
                )
            else:
                raise ValueError("Indica una ruta de archivo o sube un dataframe local.")

            return self.materialize_from_files(
                df_source,
                timestamp_col=timestamp_widget.value,
                target_cols=list(target_widget.value),
                feature_cols=feature_cols,
                entity_id_col=entity_id_col,
                train_idx_source=file_train_idx_path_widget.value.strip() or None,
                valid_idx_source=file_valid_idx_path_widget.value.strip() or None,
                test_idx_source=file_test_idx_path_widget.value.strip() or None,
                split_col=None if file_split_mode_widget.value != "split_col" or split_col_widget.value == "__none__" else split_col_widget.value,
                train_value=train_label_widget.value,
                valid_value=valid_label_widget.value,
                test_value=test_label_widget.value,
                spec_kwargs=spec_kwargs,
            )

        def _render(_=None) -> None:
            with status_output:
                clear_output(wait=True)
                print("Preparando configuracion interactiva de Eden...")

            try:
                result = _build_from_widgets()
                self.last_result_ = result
                state["result"] = result
                state["run_result"] = None
                state["config_frame"] = _build_config_frame(result)
                _sync_split_widgets()
                _render_views()

                _eden_publish_notebook_bindings(
                    eden_workbench=self,
                    eden_workbench_result=result,
                    eden_source_df=result.df,
                    eden_spec=result.spec,
                    eden_splits=result.splits,
                    eden_partitions=result.partitions,
                )

                with status_output:
                    clear_output(wait=True)
                    print("Eden interactivo preparado. Variables publicadas: eden_spec, eden_splits, eden_source_df, eden_partitions.")
            except Exception as exc:
                with summary_output:
                    clear_output(wait=True)
                with table_output:
                    clear_output(wait=True)
                with status_output:
                    clear_output(wait=True)
                    print(f"No se pudo preparar Eden: {exc}")

        def _train(_=None) -> None:
            with status_output:
                clear_output(wait=True)
                print("Entrenando el modelo seleccionado desde la tabla de Eden...")

            try:
                if state["result"] is None:
                    result = _build_from_widgets()
                    self.last_result_ = result
                    state["result"] = result
                    state["config_frame"] = _build_config_frame(result)
                    _sync_split_widgets()

                run_result = self.train_and_predict(
                    runtime=runtime_widget.value,
                    prediction_split=prediction_split_widget.value,
                    namespace_model_name=None if namespace_model_widget.value == "__none__" else namespace_model_widget.value,
                    namespace_adapter_name=None if namespace_adapter_widget.value == "__none__" else namespace_adapter_widget.value,
                    include_train_metrics=bool(include_train_metrics_widget.value),
                    interval_coverage=float(interval_coverage_widget.value),
                )
                state["run_result"] = run_result
                view_widget.value = "predictions"
                _render_views()

                _eden_publish_notebook_bindings(
                    eden_model=run_result.eden_model,
                    eden_run_result=run_result,
                    eden_predictions=run_result.selected_predictions,
                    eden_metrics=run_result.selected_metrics,
                    eden_performance_report=run_result.performance_report,
                )

                with status_output:
                    clear_output(wait=True)
                    print(
                        "Modelo entrenado. Variables publicadas: eden_model, eden_run_result, eden_predictions, eden_metrics, eden_performance_report."
                    )
            except Exception as exc:
                with status_output:
                    clear_output(wait=True)
                    print(f"No se pudo entrenar el modelo desde la tabla: {exc}")

        source_widget.observe(_sync_columns, names="value")
        bundle_widget.observe(_sync_columns, names="value")
        original_df_widget.observe(_sync_columns, names="value")
        df_widget.observe(_sync_columns, names="value")
        variable_split_mode_widget.observe(_sync_columns, names="value")
        file_split_mode_widget.observe(_sync_columns, names="value")
        file_df_path_widget.observe(_sync_columns, names="value")
        file_df_upload_widget.observe(_sync_columns, names="value")
        timestamp_widget.observe(_sync_columns, names="value")
        entity_widget.observe(_sync_columns, names="value")
        target_widget.observe(_sync_columns, names="value")
        runtime_widget.observe(_sync_runtime_controls, names="value")
        view_widget.observe(_render_views, names="value")
        preview_split_widget.observe(_render_views, names="value")
        preview_rows_widget.observe(_render_views, names="value")
        refresh_button.on_click(_render)
        train_button.on_click(_train)

        _sync_columns()
        _render()

        controls = widgets.VBox(
            [
                widgets.HTML("<b>Tabla interactiva de Eden</b>"),
                widgets.HTML(
                    "<i>Selecciona un bundle Bahamut, variables del notebook o archivos locales para construir df, spec y splits temporales listos para forecasting.</i>"
                ),
                source_widget,
                bahamut_box,
                variables_box,
                files_box,
                widgets.HBox([timestamp_widget, entity_widget]),
                widgets.HBox([target_widget, feature_widget]),
                widgets.HBox([runtime_widget, prediction_split_widget]),
                widgets.HBox([interval_coverage_widget, include_train_metrics_widget]),
                namespace_model_box,
                namespace_adapter_box,
                widgets.HBox([view_widget, preview_split_widget, preview_rows_widget]),
                widgets.HBox([refresh_button, train_button]),
                status_output,
            ],
            layout=widgets.Layout(width="760px"),
        )
        content = widgets.VBox([summary_output, table_output], layout=widgets.Layout(width="100%"))
        return widgets.HBox([controls, content], layout=widgets.Layout(align_items="flex-start"))

    interactive_table = tabla_interactiva

    def _require_dataframe(self, name: str) -> pd.DataFrame:
        if name not in self.namespace:
            raise KeyError(f"No existe la variable '{name}' en el namespace del notebook.")
        value = self.namespace[name]
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"La variable '{name}' no es un pandas.DataFrame.")
        return value.copy()

    def _require_indexer(self, name: str | None, role: str) -> list[Any]:
        if name is None:
            raise ValueError(f"Selecciona la variable {role}_idx.")
        if name not in self.namespace:
            raise KeyError(f"No existe la variable '{name}' en el namespace del notebook.")
        value = self.namespace[name]
        if not _eden_is_indexer_candidate(value):
            raise TypeError(f"La variable '{name}' no es una secuencia valida de indices.")
        return list(value)

    def _require_optional_indexer(self, name: str | None) -> list[Any] | None:
        if name is None:
            return None
        return self._require_indexer(name, role=name)

    def _build_result(
        self,
        *,
        df: pd.DataFrame,
        splits: EdenSplits,
        timestamp_col: str,
        target_cols: Sequence[str],
        feature_cols: Sequence[str] | None,
        entity_id_col: str | None,
        source_mode: str,
        source_name: str | None,
        metadata: Mapping[str, Any] | None = None,
        spec_kwargs: Mapping[str, Any] | None = None,
    ) -> EdenWorkbenchResult:
        resolved_feature_cols = list(feature_cols) if feature_cols is not None else [
            column
            for column in df.columns
            if column not in set(target_cols) | {timestamp_col, entity_id_col}
        ]
        series_structure = "panel" if entity_id_col is not None else "single"
        spec = EdenSpec(
            timestamp_col=timestamp_col,
            target_cols=list(target_cols),
            feature_cols=resolved_feature_cols,
            series_structure=series_structure,
            entity_id_col=entity_id_col,
            **dict(spec_kwargs or {}),
        )
        validator = Eden(spec=spec, model=_PreviewEstimator())
        partitions = validator.partitions(df, splits, require_targets=False)
        return EdenWorkbenchResult(
            source_mode=source_mode,
            source_name=source_name,
            df=df.copy(),
            splits=splits,
            spec=spec,
            partitions=partitions,
            metadata=dict(metadata or {}),
        )

    def _build_file_splits(
        self,
        df: pd.DataFrame,
        split_mode: str,
        split_col: str,
        train_value: str,
        valid_value: str,
        test_value: str,
        train_path: str,
        valid_path: str,
        test_path: str,
        train_upload_value: Any,
        valid_upload_value: Any,
        test_upload_value: Any,
    ) -> EdenSplits:
        if split_mode == "split_col":
            if split_col == "__none__":
                raise ValueError("Selecciona una columna split para el dataframe cargado.")
            return _eden_build_splits_from_column(
                df,
                split_col=split_col,
                train_value=train_value,
                valid_value=valid_value,
                test_value=test_value,
            )

        train_idx = self._load_index_source(train_path, train_upload_value, role="train")
        valid_idx = self._load_index_source(valid_path, valid_upload_value, role="valid", required=False)
        test_idx = self._load_index_source(test_path, test_upload_value, role="test", required=False)
        return EdenSplits(train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

    def _load_index_source(self, path: str, upload_value: Any, *, role: str, required: bool = True) -> list[Any] | None:
        if path.strip():
            return _eden_read_indexer_from_path(path.strip())

        uploaded = _eden_first_upload(upload_value)
        if uploaded is not None:
            filename, content = uploaded
            return _eden_read_indexer_from_buffer(content, filename)

        if required:
            raise ValueError(f"Debes indicar un archivo de indices para {role}.")
        return None


def tabla_interactiva(namespace: Mapping[str, Any] | None = None, preview_rows: int = 6):
    return EdenWorkbench(namespace=namespace).tabla_interactiva(preview_rows=preview_rows)


__all__ = ["EdenWorkbench", "EdenWorkbenchResult", "EdenWorkbenchRunResult", "tabla_interactiva"]