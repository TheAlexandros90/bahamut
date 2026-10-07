from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# La capa interactiva vive en bahamut/interactive.py y se engancha a
# BahamutSplit desde bahamut/__init__.py. Este modulo es solo el nucleo
# analitico: no importa ipywidgets ni IPython.

ProblemaML = Literal[
    "regresion",
    "clasificacion",
    "series_temporales",
    "clusterizacion",
    "otro",
]
ColumnasInput = Union[str, List[str], Tuple[str, ...]]


@dataclass(frozen=True)
class BahamutSplitConfig:
    """Configuracion validada para ejecutar un split reproducible y de calidad."""

    problema: ProblemaML
    target_cols: Tuple[str, ...]
    feature_cols: Tuple[str, ...]
    exclude_cols: Tuple[str, ...]
    test_size: float
    validation_size: Optional[float]
    train_size: float
    random_state: Optional[int]
    shuffle: bool
    stratify_by: Optional[Union[str, List[str]]]
    stratify_mode: str
    regression_bins: Optional[int]
    time_col: Optional[str]
    time_ascending: bool
    group_cols: Tuple[str, ...]
    use_group_split: bool
    split_method: str


class Bahamut:
    """Superclase base para futuras utilidades de machine learning dentro de Bahamut."""

    PROBLEMAS_VALIDOS = {
        "regresion",
        "clasificacion",
        "series_temporales",
        "clusterizacion",
        "otro",
    }

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame")
        if df.empty:
            raise ValueError("df no puede estar vacio")

        self.df = df.copy()

    def columnas_disponibles(self) -> List[str]:
        return self.df.columns.tolist()

    def resumen_dataset(self) -> Dict[str, Any]:
        return {
            "n_filas": int(self.df.shape[0]),
            "n_columnas": int(self.df.shape[1]),
            "columnas": self.columnas_disponibles(),
        }

    def _normalizar_columnas(
        self,
        cols: Optional[ColumnasInput],
        nombre_argumento: str,
        permitir_vacio: bool = False,
    ) -> List[str]:
        if cols is None:
            return []

        columnas = [cols] if isinstance(cols, str) else list(cols)
        columnas = [str(col).strip() for col in columnas if str(col).strip()]

        if not columnas and not permitir_vacio:
            raise ValueError(f"{nombre_argumento} no puede estar vacio")

        self._validar_columnas(columnas, nombre_argumento=nombre_argumento)
        return columnas

    def _normalizar_columna_unica(self, col: Optional[str], nombre_argumento: str) -> Optional[str]:
        if col is None:
            return None

        columna = str(col).strip()
        if not columna:
            return None

        self._validar_columnas([columna], nombre_argumento=nombre_argumento)
        return columna

    def _validar_columnas(self, cols: List[str], nombre_argumento: str) -> None:
        inexistentes = [col for col in cols if col not in self.df.columns]
        if inexistentes:
            raise ValueError(
                f"Columnas no encontradas en df para {nombre_argumento}: {inexistentes}"
            )

    def _colapsar_columnas_en_serie(self, frame: pd.DataFrame, cols: List[str]) -> pd.Series:
        if len(cols) == 1:
            return frame[cols[0]].copy()
        return frame[cols].astype(str).agg("|".join, axis=1)


class BahamutSplit(Bahamut):
    """
    Subclase de Bahamut especializada en construir splits robustos, configurables e interactivos.

    Cobertura principal:
    1. Holdout clasico train/test.
    2. Segmentacion train/validation/test.
    3. Estratificacion por target o por columnas.
    4. Tratamiento especifico para series temporales.
    5. Segmentacion por grupos para evitar leakage.
    6. Tabla interactiva y diagnostico de variables configurables.
    """

    MODOS_ESTRATIFICACION_VALIDOS = {"auto", "target", "columnas", "ninguna"}
    # Modos en los que el usuario pide estratificar de forma explicita. "auto"
    # queda fuera a proposito: es una preferencia, no un requisito.
    MODOS_ESTRATIFICACION_EXPLICITOS = {"target", "columnas"}

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.problema: Optional[ProblemaML] = None
        self.target_cols: List[str] = []
        self.feature_cols: List[str] = []
        self.exclude_cols: List[str] = []
        self.stratify_cols: List[str] = []
        self.group_cols: List[str] = []
        self.stratify_mode: str = "auto"
        self.regression_bins: Optional[int] = 5
        self.shuffle: bool = True
        self.test_size: float = 0.2
        self.validation_size: Optional[float] = None
        self.train_size: Optional[float] = None
        self.random_state: Optional[int] = 42
        self.time_col: Optional[str] = None
        self.time_ascending: bool = True
        self.asumir_orden_actual: bool = False

    def definir_problema(self, problema: str) -> "BahamutSplit":
        problema_norm = str(problema).strip().lower()
        if problema_norm not in self.PROBLEMAS_VALIDOS:
            raise ValueError(
                f"Problema '{problema}' no valido. Usa uno de: {sorted(self.PROBLEMAS_VALIDOS)}"
            )
        self.problema = problema_norm
        return self

    def definir_objetivo(self, target_cols: Optional[ColumnasInput]) -> "BahamutSplit":
        if target_cols is None:
            self.target_cols = []
            return self

        self.target_cols = self._normalizar_columnas(
            target_cols,
            nombre_argumento="target_cols",
        )
        return self

    def definir_predictoras(
        self,
        feature_cols: Optional[ColumnasInput] = None,
        exclude_cols: Optional[ColumnasInput] = None,
    ) -> "BahamutSplit":
        self.feature_cols = self._normalizar_columnas(
            feature_cols,
            nombre_argumento="feature_cols",
            permitir_vacio=True,
        ) if feature_cols is not None else []

        self.exclude_cols = self._normalizar_columnas(
            exclude_cols,
            nombre_argumento="exclude_cols",
            permitir_vacio=True,
        ) if exclude_cols is not None else []

        overlap = sorted(set(self.feature_cols) & set(self.exclude_cols))
        if overlap:
            raise ValueError(f"feature_cols y exclude_cols no deben solaparse: {overlap}")

        return self

    def definir_orden_temporal(
        self,
        time_col: Optional[str],
        ascending: bool = True,
        *,
        asumir_orden_actual: bool = False,
    ) -> "BahamutSplit":
        """Define como se ordena el DataFrame antes de una segmentacion temporal.

        Para `problema="series_temporales"` hay que indicar `time_col`. Sin ella,
        el corte se haria sobre el orden actual de las filas, que puede no ser
        cronologico: eso mete el futuro en train y el pasado en test sin avisar.
        Si de verdad sabes que el DataFrame ya viene ordenado, pide ese
        comportamiento de forma explicita con `asumir_orden_actual=True`.
        """
        self.time_col = self._normalizar_columna_unica(time_col, nombre_argumento="time_col")
        self.time_ascending = bool(ascending)
        self.asumir_orden_actual = bool(asumir_orden_actual)
        return self

    def definir_estratificacion(
        self,
        stratify_cols: Optional[ColumnasInput] = None,
        *,
        modo: str = "auto",
        bins_regresion: Optional[int] = 5,
    ) -> "BahamutSplit":
        # Toda la validacion ocurre antes de tocar el estado del objeto: si algo
        # falla, la instancia queda exactamente como estaba.
        modo_norm = str(modo).strip().lower()
        if modo_norm not in self.MODOS_ESTRATIFICACION_VALIDOS:
            raise ValueError(
                "modo debe ser uno de: "
                f"{sorted(self.MODOS_ESTRATIFICACION_VALIDOS)}"
            )

        if modo_norm != "columnas" and stratify_cols is not None:
            raise ValueError("Solo puedes pasar stratify_cols cuando modo='columnas'.")

        bins = None if bins_regresion is None else int(bins_regresion)
        if bins is not None and bins < 2:
            raise ValueError("bins_regresion debe ser al menos 2")

        columnas_validadas: List[str] = []
        if modo_norm == "columnas":
            columnas_validadas = self._normalizar_columnas(
                stratify_cols,
                nombre_argumento="stratify_cols",
            )

        self.stratify_mode = modo_norm
        self.regression_bins = bins
        self.stratify_cols = columnas_validadas
        return self

    def definir_grupos(self, group_cols: Optional[ColumnasInput]) -> "BahamutSplit":
        self.group_cols = self._normalizar_columnas(
            group_cols,
            nombre_argumento="group_cols",
            permitir_vacio=True,
        ) if group_cols is not None else []
        return self

    def configurar_split(
        self,
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        train_size: Optional[float] = None,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> "BahamutSplit":
        test_size = float(test_size)
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1")

        if validation_size is not None:
            validation_size = float(validation_size)
            if not 0 < validation_size < 1:
                raise ValueError("validation_size debe estar entre 0 y 1")

        if train_size is not None:
            train_size = float(train_size)
            if not 0 < train_size < 1:
                raise ValueError("train_size debe estar entre 0 y 1")

        if validation_size is None and train_size is not None:
            total = train_size + test_size
            if abs(total - 1.0) > 1e-9:
                raise ValueError("Si indicas train_size y test_size, deben sumar 1.")

        if validation_size is not None and train_size is not None:
            total = train_size + test_size + validation_size
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    "Si indicas train_size, validation_size y test_size, deben sumar 1."
                )

        if validation_size is not None and train_size is None and (test_size + validation_size) >= 1:
            raise ValueError("test_size + validation_size debe ser menor que 1")

        if self.problema == "series_temporales" and shuffle:
            raise ValueError(
                "Para series_temporales, un split de calidad exige shuffle=False."
            )

        if self.group_cols and self.problema == "series_temporales":
            raise ValueError(
                "Bahamut no combina group split y series_temporales. Usa time_col para una segmentacion temporal explicita."
            )

        # `auto` significa "decide tu": no puede provocar un error de
        # configuracion. Solo se rechazan las peticiones explicitas de
        # estratificar que son incompatibles con el resto de la configuracion.
        if self.group_cols and self.stratify_mode in self.MODOS_ESTRATIFICACION_EXPLICITOS:
            raise ValueError(
                "Bahamut no combina group split y estratificacion en esta version. Usa uno u otro."
            )

        if not shuffle and self.stratify_mode in self.MODOS_ESTRATIFICACION_EXPLICITOS:
            raise ValueError(
                "No se puede estratificar si shuffle=False en train_test_split. "
                "Usa modo='ninguna' o activa shuffle."
            )

        self.test_size = test_size
        self.validation_size = validation_size
        self.train_size = train_size
        self.shuffle = bool(shuffle)
        self.random_state = None if not shuffle else None if random_state is None else int(random_state)
        return self

    def configurar_segmentacion(self, **kwargs: Any) -> "BahamutSplit":
        return self.configurar_split(**kwargs)

    def variables_disponibles_para_split(self) -> Dict[str, List[str]]:
        """Enumera las columnas disponibles y como pueden intervenir en el split."""
        columnas = self.columnas_disponibles()
        sin_target = [col for col in columnas if col not in self.target_cols]

        try:
            feature_cols_resueltas = self._resolver_feature_cols()
        except ValueError:
            feature_cols_resueltas = []

        return {
            "columnas_totales": columnas,
            "columnas_candidatas_target": columnas,
            "columnas_candidatas_feature": sin_target,
            "columnas_candidatas_exclude": sin_target,
            "columnas_candidatas_stratify": sin_target,
            "columnas_candidatas_group": columnas,
            "columnas_candidatas_time": columnas,
            "target_cols_actuales": list(self.target_cols),
            "feature_cols_configuradas": list(self.feature_cols),
            "feature_cols_resueltas": feature_cols_resueltas,
            "exclude_cols_actuales": list(self.exclude_cols),
            "stratify_cols_actuales": list(self.stratify_cols),
            "group_cols_actuales": list(self.group_cols),
            "time_col_actual": [] if self.time_col is None else [self.time_col],
        }

    def tabla_variables_split(self) -> pd.DataFrame:
        """Devuelve una vista tabular de todas las variables que participan o pueden participar en el split."""
        rows: List[Dict[str, Any]] = []
        for categoria, columnas in self.variables_disponibles_para_split().items():
            rows.append(
                {
                    "categoria": categoria,
                    "n_columnas": int(len(columnas)),
                    "columnas": ", ".join(columnas),
                }
            )
        return pd.DataFrame(rows)

    def obtener_configuracion(self) -> BahamutSplitConfig:
        self._validar_estado_minimo()
        return self._construir_config_split()

    def parametros_para_split_manual(self) -> Dict[str, Any]:
        """Devuelve los parametros listos para reproducir manualmente el split."""
        config = self.obtener_configuracion()
        frame = self._preparar_frame_base(config)
        _, stratify_meta = self._resolver_estratificacion(config, frame)

        parametros: Dict[str, Any] = {
            "segmentacion": "train/validation/test" if config.validation_size is not None else "train/test",
            "problema": config.problema,
            "feature_cols": list(config.feature_cols),
            "exclude_cols": list(config.exclude_cols),
            "target_cols": list(config.target_cols),
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "train_size": config.train_size,
            "random_state": config.random_state,
            "shuffle": config.shuffle,
            "stratify_mode": config.stratify_mode,
            "group_cols": list(config.group_cols),
            "use_group_split": config.use_group_split,
            "split_method": config.split_method,
        }

        if config.time_col is not None:
            parametros["time_col"] = config.time_col
            parametros["time_ascending"] = config.time_ascending
            parametros["nota_temporal"] = "Ordena el DataFrame por time_col antes de segmentar."
        elif config.problema == "series_temporales":
            parametros["time_col"] = None
            parametros["nota_temporal"] = (
                "Sin time_col: el corte usa el orden actual de las filas. "
                "Reproducirlo exige entregar el DataFrame en ese mismo orden."
            )

        if stratify_meta is not None:
            parametros["stratify"] = stratify_meta["descripcion"]
            parametros["stratify_source"] = stratify_meta["source"]
            parametros["stratify_cols"] = stratify_meta["columns"]

        if config.use_group_split:
            parametros["nota_grupos"] = (
                "La segmentacion por grupos prioriza que una misma entidad no aparezca en mas de un bloque. "
                "Las proporciones finales por filas pueden ser aproximadas."
            )

        if config.validation_size is not None:
            parametros["nota_segmentacion"] = (
                "La segmentacion a 3 bloques se resuelve con dos splits secuenciales: "
                "primero se aisla test y luego validation sobre el bloque restante."
            )

        return parametros

    def diagnostico_split(self) -> Dict[str, Any]:
        """Resumen operativo para revisar la calidad del split antes de ejecutarlo."""
        config = self.obtener_configuracion()
        frame = self._preparar_frame_base(config)
        stratify_data, stratify_meta = self._resolver_estratificacion(config, frame)
        group_data = self._resolver_grupos(config, frame)

        diagnostico: Dict[str, Any] = {
            "problema": config.problema,
            "segmentacion": "train/validation/test" if config.validation_size is not None else "train/test",
            "n_filas": int(frame.shape[0]),
            "n_features_entrada": int(len(config.feature_cols)),
            "feature_cols": list(config.feature_cols),
            "target_cols": list(config.target_cols),
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "train_size": config.train_size,
            "shuffle": config.shuffle,
            "random_state": config.random_state,
            "metodo_split": config.split_method,
            "estratificacion_activada": stratify_data is not None,
            "modo_estratificacion": config.stratify_mode,
            "modo_estratificacion_efectivo": (
                "aplicada" if stratify_data is not None else self._motivo_sin_estratificacion(config)
            ),
            "regression_bins": config.regression_bins,
            "group_split_activado": group_data is not None,
            "group_cols": list(config.group_cols),
            "time_col": config.time_col,
            "orden_temporal": self._descripcion_orden_temporal(config),
        }

        if stratify_meta is not None:
            diagnostico["estratificacion_resuelta"] = stratify_meta["descripcion"]
            diagnostico["estratificacion_source"] = stratify_meta["source"]
            diagnostico["estratificacion_cols"] = stratify_meta["columns"]
            diagnostico["n_grupos_estratificacion"] = int(pd.Series(stratify_data).nunique(dropna=False))

        if group_data is not None:
            diagnostico["n_grupos"] = int(group_data.nunique(dropna=False))

        return diagnostico

    def ejecutar_split(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[Union[pd.Series, pd.DataFrame]],
        Optional[Union[pd.Series, pd.DataFrame]],
    ]:
        """Ejecuta el split clasico train/test. Si hay validation_size, usa ejecutar_segmentacion()."""
        if self.validation_size is not None:
            raise ValueError(
                "Hay validation_size configurado. Usa ejecutar_segmentacion() para obtener train/validation/test."
            )

        segmentos = self.ejecutar_segmentacion()
        return segmentos["X_train"], segmentos["X_test"], segmentos["y_train"], segmentos["y_test"]

    def ejecutar_segmentacion(self) -> Dict[str, Any]:
        """Ejecuta la segmentacion actual y devuelve un diccionario con todos los bloques."""
        config = self.obtener_configuracion()
        frame = self._preparar_frame_base(config)
        X, y = self._resolver_X_y(frame, config)
        stratify_data, _ = self._resolver_estratificacion(config, frame)
        group_data = self._resolver_grupos(config, frame)

        if config.problema == "series_temporales":
            segmentos = self._segmentar_temporal(X, y, config)
        elif group_data is not None:
            segmentos = self._segmentar_por_grupos(X, y, group_data, config)
        else:
            segmentos = self._segmentar_aleatorio(X, y, stratify_data, config)

        segmentos["resumen_segmentos"] = self.resumen_segmentos(segmentos)
        segmentos["split_config"] = self.resumen_configuracion()
        return segmentos

    def resumen_configuracion(self) -> Dict[str, Any]:
        config = self.obtener_configuracion()
        return {
            "problema": config.problema,
            "segmentacion": "train/validation/test" if config.validation_size is not None else "train/test",
            "target_cols": list(config.target_cols),
            "feature_cols": list(config.feature_cols),
            "exclude_cols": list(config.exclude_cols),
            "stratify_mode": config.stratify_mode,
            "stratify_cols": list(self.stratify_cols),
            "group_cols": list(config.group_cols),
            "use_group_split": config.use_group_split,
            "shuffle": config.shuffle,
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "train_size": config.train_size,
            "random_state": config.random_state,
            "regression_bins": config.regression_bins,
            "time_col": config.time_col,
            "split_method": config.split_method,
        }

    def resumen_segmentos(self, segmentos: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if segmentos is None:
            segmentos = self.ejecutar_segmentacion()

        segment_keys = [
            ("train", "X_train", "y_train", "groups_train"),
            ("validation", "X_validation", "y_validation", "groups_validation"),
            ("test", "X_test", "y_test", "groups_test"),
        ]
        total_rows = sum(
            len(segmentos[x_key])
            for _, x_key, _, _ in segment_keys
            if segmentos.get(x_key) is not None
        )

        rows: List[Dict[str, Any]] = []
        for segment_name, x_key, y_key, group_key in segment_keys:
            X_part = segmentos.get(x_key)
            if X_part is None:
                continue

            y_part = segmentos.get(y_key)
            groups_part = segmentos.get(group_key)
            row: Dict[str, Any] = {
                "segmento": segment_name,
                "filas": int(len(X_part)),
                "proporcion": float(len(X_part) / total_rows) if total_rows else 0.0,
                "columnas_X": int(X_part.shape[1]),
            }

            if y_part is None:
                row["target_tipo"] = "sin target"
                row["target_unicos"] = None
            elif isinstance(y_part, pd.Series):
                row["target_tipo"] = "serie"
                row["target_unicos"] = int(y_part.nunique(dropna=False))
            else:
                row["target_tipo"] = f"dataframe({y_part.shape[1]})"
                row["target_unicos"] = None

            if self.group_cols:
                row["grupos_unicos"] = None if groups_part is None else int(groups_part.nunique(dropna=False))

            rows.append(row)

        return pd.DataFrame(rows)

    def balance_target_por_segmento(
        self,
        segmentos: Optional[Dict[str, Any]] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        if not self.target_cols:
            raise ValueError("No hay target definido para construir un balance por segmento.")
        if len(self.target_cols) != 1:
            raise ValueError("El balance interactivo solo esta soportado para un target simple.")

        if segmentos is None:
            segmentos = self.ejecutar_segmentacion()

        frames: List[pd.Series] = []
        for segment_name, y_key in [("train", "y_train"), ("validation", "y_validation"), ("test", "y_test")]:
            y_part = segmentos.get(y_key)
            if y_part is None:
                continue
            if not isinstance(y_part, pd.Series):
                raise ValueError("El balance por segmento requiere target de una sola columna.")
            counts = y_part.value_counts(normalize=normalize, dropna=False).rename(segment_name)
            frames.append(counts)

        if not frames:
            raise ValueError("No hay suficientes segmentos con target para construir el balance.")

        return pd.concat(frames, axis=1).fillna(0)

    def preview_segmento(
        self,
        segmentos: Dict[str, Any],
        segmento: str = "train",
        rows: int = 5,
    ) -> pd.DataFrame:
        mapping = {
            "train": ("X_train", "y_train"),
            "validation": ("X_validation", "y_validation"),
            "test": ("X_test", "y_test"),
        }
        if segmento not in mapping:
            raise ValueError(f"segmento debe ser uno de: {sorted(mapping)}")

        x_key, y_key = mapping[segmento]
        X_part = segmentos.get(x_key)
        y_part = segmentos.get(y_key)
        if X_part is None:
            raise ValueError(f"El segmento '{segmento}' no esta disponible con la configuracion actual.")

        preview = X_part.head(int(rows)).copy()
        if y_part is None:
            return preview
        if isinstance(y_part, pd.Series):
            preview[y_part.name or "target"] = y_part.head(int(rows)).values
            return preview
        return pd.concat([preview, y_part.head(int(rows))], axis=1)


    def _construir_config_split(self) -> BahamutSplitConfig:
        feature_cols = tuple(self._resolver_feature_cols())
        effective_train_size = self._resolver_train_size()
        stratify_by: Optional[Union[str, List[str]]] = None
        if self.stratify_mode == "columnas":
            stratify_by = list(self.stratify_cols)
        elif self.stratify_mode == "target" and self.target_cols:
            stratify_by = list(self.target_cols)

        return BahamutSplitConfig(
            problema=self.problema,
            target_cols=tuple(self.target_cols),
            feature_cols=feature_cols,
            exclude_cols=tuple(self.exclude_cols),
            test_size=float(self.test_size),
            validation_size=self.validation_size,
            train_size=float(effective_train_size),
            random_state=self.random_state,
            shuffle=bool(self.shuffle),
            stratify_by=stratify_by,
            stratify_mode=self.stratify_mode,
            regression_bins=self.regression_bins,
            time_col=self.time_col,
            time_ascending=self.time_ascending,
            group_cols=tuple(self.group_cols),
            use_group_split=bool(self.group_cols),
            split_method=self._resolver_metodo_split(),
        )

    def _motivo_sin_estratificacion(self, config: BahamutSplitConfig) -> str:
        if config.stratify_mode == "ninguna":
            return "desactivada por configuracion"
        if config.use_group_split:
            return "desactivada: incompatible con group split"
        if not config.shuffle:
            return "desactivada: shuffle=False"
        if config.stratify_mode == "auto":
            return "auto no encontro una estratificacion aplicable"
        return "no aplicada"

    def _descripcion_orden_temporal(self, config: BahamutSplitConfig) -> str:
        if config.problema != "series_temporales":
            return "no aplica"
        if config.time_col is not None:
            sentido = "ascendente" if config.time_ascending else "descendente"
            return f"ordenado por '{config.time_col}' ({sentido})"
        return "orden actual de las filas del DataFrame (asumido de forma explicita)"

    def _resolver_metodo_split(self) -> str:
        if self.problema == "series_temporales":
            return "temporal_ordered"
        if self.group_cols:
            return "group_shuffle_split" if self.shuffle else "group_ordered_split"
        return "train_test_split"

    def _resolver_feature_cols(self) -> List[str]:
        base_cols = self.feature_cols if self.feature_cols else self.columnas_disponibles()
        feature_cols = [
            col for col in base_cols
            if col not in self.target_cols and col not in self.exclude_cols
        ]
        if not feature_cols:
            raise ValueError("No quedan columnas predictoras tras aplicar target_cols y exclude_cols.")
        return feature_cols

    def _resolver_train_size(self) -> float:
        if self.train_size is not None:
            return float(self.train_size)

        validation_size = 0.0 if self.validation_size is None else float(self.validation_size)
        train_size = 1.0 - float(self.test_size) - validation_size
        if train_size <= 0:
            raise ValueError("La proporcion de train queda no positiva con la configuracion actual.")
        return train_size

    def _preparar_frame_base(self, config: BahamutSplitConfig) -> pd.DataFrame:
        frame = self.df.copy()
        if config.time_col is not None:
            frame = frame.sort_values(config.time_col, ascending=config.time_ascending)
        return frame

    def _resolver_X_y(
        self,
        frame: pd.DataFrame,
        config: BahamutSplitConfig,
    ) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, pd.DataFrame]]]:
        X = frame.loc[:, list(config.feature_cols)].copy()
        if not config.target_cols:
            return X, None
        if len(config.target_cols) == 1:
            return X, frame[config.target_cols[0]].copy()
        return X, frame.loc[:, list(config.target_cols)].copy()

    def _resolver_grupos(
        self,
        config: BahamutSplitConfig,
        frame: pd.DataFrame,
    ) -> Optional[pd.Series]:
        if not config.use_group_split:
            return None

        return (
            self._colapsar_columnas_en_serie(frame, list(config.group_cols))
            .astype("string")
            .fillna("__bahamut_nan_group__")
            .astype(str)
        )

    def _resolver_estratificacion(
        self,
        config: BahamutSplitConfig,
        frame: pd.DataFrame,
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        if config.use_group_split:
            return None, None

        if not config.shuffle or config.stratify_mode == "ninguna":
            return None, None

        if config.stratify_mode == "columnas":
            stratify_series = self._colapsar_columnas_en_serie(frame, list(self.stratify_cols)).astype(str)
            return stratify_series, {
                "source": "columnas",
                "columns": list(self.stratify_cols),
                "descripcion": f"estratificacion por columnas: {self.stratify_cols}",
            }

        if config.stratify_mode == "target":
            return self._resolver_estratificacion_por_target(config, frame)

        if config.stratify_mode == "auto":
            if config.problema in {"clasificacion", "regresion"} and config.target_cols:
                return self._resolver_estratificacion_por_target(config, frame)
            return None, None

        return None, None

    def _resolver_estratificacion_por_target(
        self,
        config: BahamutSplitConfig,
        frame: pd.DataFrame,
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        if not config.target_cols:
            raise ValueError("No puedes estratificar por target si no has definido target_cols.")

        if config.problema == "regresion":
            if len(config.target_cols) != 1:
                raise ValueError(
                    "La estratificacion por target en regresion requiere una sola variable objetivo numerica."
                )

            target_name = config.target_cols[0]
            target_series = frame[target_name].copy()
            if not is_numeric_dtype(target_series):
                raise ValueError(
                    "Para regresion, la estratificacion por target requiere una variable objetivo numerica."
                )

            bins = 5 if config.regression_bins is None else int(config.regression_bins)
            bins = min(bins, int(target_series.nunique(dropna=True)))
            if bins < 2:
                return None, None

            stratify_series = pd.qcut(target_series, q=bins, duplicates="drop").astype(str)
            return stratify_series, {
                "source": "target_bins",
                "columns": [target_name],
                "descripcion": f"estratificacion por target binned en {bins} cuantiles",
            }

        if len(config.target_cols) == 1:
            target_name = config.target_cols[0]
            return frame[target_name].copy(), {
                "source": "target",
                "columns": [target_name],
                "descripcion": f"estratificacion por target: {target_name}",
            }

        stratify_series = self._colapsar_columnas_en_serie(frame, list(config.target_cols)).astype(str)
        return stratify_series, {
            "source": "multi_target",
            "columns": list(config.target_cols),
            "descripcion": f"estratificacion por combinacion de targets: {list(config.target_cols)}",
        }

    def _construir_segmentos(
        self,
        *,
        X_train: pd.DataFrame,
        X_validation: Optional[pd.DataFrame],
        X_test: pd.DataFrame,
        y_train: Optional[Union[pd.Series, pd.DataFrame]],
        y_validation: Optional[Union[pd.Series, pd.DataFrame]],
        y_test: Optional[Union[pd.Series, pd.DataFrame]],
        groups_train: Optional[pd.Series] = None,
        groups_validation: Optional[pd.Series] = None,
        groups_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        return {
            "X_train": X_train,
            "X_validation": X_validation,
            "X_test": X_test,
            "y_train": y_train,
            "y_validation": y_validation,
            "y_test": y_test,
            "groups_train": groups_train,
            "groups_validation": groups_validation,
            "groups_test": groups_test,
            "indices_train": X_train.index.copy(),
            "indices_validation": None if X_validation is None else X_validation.index.copy(),
            "indices_test": X_test.index.copy(),
        }

    def _segmentar_temporal(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        n_rows = len(X)
        n_test = max(1, int(round(n_rows * config.test_size)))
        n_validation = 0 if config.validation_size is None else max(1, int(round(n_rows * config.validation_size)))
        n_train = n_rows - n_test - n_validation

        if n_train < 1:
            raise ValueError("No quedan filas suficientes para train en la segmentacion temporal.")

        train_end = n_train
        valid_end = n_train + n_validation

        X_train = X.iloc[:train_end].copy()
        X_validation = X.iloc[train_end:valid_end].copy() if n_validation else None
        X_test = X.iloc[valid_end:].copy()

        if y is None:
            y_train = None
            y_validation = None
            y_test = None
        else:
            y_train = y.iloc[:train_end].copy()
            y_validation = y.iloc[train_end:valid_end].copy() if n_validation else None
            y_test = y.iloc[valid_end:].copy()

        return self._construir_segmentos(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )

    def _segmentar_aleatorio(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        stratify_data: Optional[pd.Series],
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        try:
            if config.validation_size is None:
                return self._segmentar_holdout(X, y, stratify_data, config)
            return self._segmentar_tripartito(X, y, stratify_data, config)
        except ValueError as exc:
            raise ValueError(f"No se pudo ejecutar la segmentacion con la configuracion actual: {exc}") from exc

    def _segmentar_holdout(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        stratify_data: Optional[pd.Series],
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        if y is None:
            if stratify_data is None:
                X_train, X_test = train_test_split(
                    X,
                    test_size=config.test_size,
                    train_size=config.train_size,
                    random_state=config.random_state,
                    shuffle=config.shuffle,
                )
            else:
                X_train, X_test, _, _ = train_test_split(
                    X,
                    stratify_data,
                    test_size=config.test_size,
                    train_size=config.train_size,
                    random_state=config.random_state,
                    shuffle=config.shuffle,
                    stratify=stratify_data,
                )
            return self._construir_segmentos(
                X_train=X_train,
                X_validation=None,
                X_test=X_test,
                y_train=None,
                y_validation=None,
                y_test=None,
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            train_size=config.train_size,
            random_state=config.random_state,
            shuffle=config.shuffle,
            stratify=stratify_data,
        )
        return self._construir_segmentos(
            X_train=X_train,
            X_validation=None,
            X_test=X_test,
            y_train=y_train,
            y_validation=None,
            y_test=y_test,
        )

    def _segmentar_tripartito(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        stratify_data: Optional[pd.Series],
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        validation_ratio = float(config.validation_size) / (1.0 - float(config.test_size))
        second_random_state = None if config.random_state is None else int(config.random_state) + 1

        if y is None:
            if stratify_data is None:
                X_temp, X_test = train_test_split(
                    X,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    shuffle=config.shuffle,
                )
                X_train, X_validation = train_test_split(
                    X_temp,
                    test_size=validation_ratio,
                    random_state=second_random_state,
                    shuffle=config.shuffle,
                )
            else:
                X_temp, X_test, stratify_temp, _ = train_test_split(
                    X,
                    stratify_data,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    shuffle=config.shuffle,
                    stratify=stratify_data,
                )
                X_train, X_validation, _, _ = train_test_split(
                    X_temp,
                    stratify_temp,
                    test_size=validation_ratio,
                    random_state=second_random_state,
                    shuffle=config.shuffle,
                    stratify=stratify_temp,
                )

            return self._construir_segmentos(
                X_train=X_train,
                X_validation=X_validation,
                X_test=X_test,
                y_train=None,
                y_validation=None,
                y_test=None,
            )

        if stratify_data is None:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                shuffle=config.shuffle,
            )
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_temp,
                y_temp,
                test_size=validation_ratio,
                random_state=second_random_state,
                shuffle=config.shuffle,
            )
        else:
            X_temp, X_test, y_temp, y_test, stratify_temp, _ = train_test_split(
                X,
                y,
                stratify_data,
                test_size=config.test_size,
                random_state=config.random_state,
                shuffle=config.shuffle,
                stratify=stratify_data,
            )
            X_train, X_validation, y_train, y_validation, _, _ = train_test_split(
                X_temp,
                y_temp,
                stratify_temp,
                test_size=validation_ratio,
                random_state=second_random_state,
                shuffle=config.shuffle,
                stratify=stratify_temp,
            )

        return self._construir_segmentos(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )

    def _segmentar_por_grupos(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        group_data: pd.Series,
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        try:
            if config.validation_size is None:
                return self._segmentar_grupos_holdout(X, y, group_data, config)
            return self._segmentar_grupos_tripartito(X, y, group_data, config)
        except ValueError as exc:
            raise ValueError(f"No se pudo ejecutar la segmentacion por grupos con la configuracion actual: {exc}") from exc

    def _segmentar_grupos_holdout(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        group_data: pd.Series,
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        if config.shuffle:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.test_size,
                train_size=config.train_size,
                random_state=config.random_state,
            )
            train_pos, test_pos = next(splitter.split(X, y, groups=group_data))
            X_train = X.iloc[train_pos].copy()
            X_test = X.iloc[test_pos].copy()
            groups_train = group_data.iloc[train_pos].copy()
            groups_test = group_data.iloc[test_pos].copy()
            if y is None:
                y_train = None
                y_test = None
            else:
                y_train = y.iloc[train_pos].copy()
                y_test = y.iloc[test_pos].copy()
        else:
            train_mask, _, test_mask = self._resolver_mascaras_grupos_ordenados(group_data, config)
            X_train = X.loc[train_mask].copy()
            X_test = X.loc[test_mask].copy()
            groups_train = group_data.loc[train_mask].copy()
            groups_test = group_data.loc[test_mask].copy()
            if y is None:
                y_train = None
                y_test = None
            else:
                y_train = y.loc[train_mask].copy()
                y_test = y.loc[test_mask].copy()

        return self._construir_segmentos(
            X_train=X_train,
            X_validation=None,
            X_test=X_test,
            y_train=y_train,
            y_validation=None,
            y_test=y_test,
            groups_train=groups_train,
            groups_validation=None,
            groups_test=groups_test,
        )

    def _segmentar_grupos_tripartito(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        group_data: pd.Series,
        config: BahamutSplitConfig,
    ) -> Dict[str, Any]:
        if config.shuffle:
            validation_ratio = float(config.validation_size) / (1.0 - float(config.test_size))
            second_random_state = None if config.random_state is None else int(config.random_state) + 1
            first_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.test_size,
                random_state=config.random_state,
            )
            temp_pos, test_pos = next(first_splitter.split(X, y, groups=group_data))

            X_temp = X.iloc[temp_pos].copy()
            X_test = X.iloc[test_pos].copy()
            groups_temp = group_data.iloc[temp_pos].copy()
            groups_test = group_data.iloc[test_pos].copy()

            if y is None:
                y_temp = None
                y_test = None
            else:
                y_temp = y.iloc[temp_pos].copy()
                y_test = y.iloc[test_pos].copy()

            second_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_ratio,
                random_state=second_random_state,
            )
            train_pos_rel, validation_pos_rel = next(second_splitter.split(X_temp, y_temp, groups=groups_temp))

            X_train = X_temp.iloc[train_pos_rel].copy()
            X_validation = X_temp.iloc[validation_pos_rel].copy()
            groups_train = groups_temp.iloc[train_pos_rel].copy()
            groups_validation = groups_temp.iloc[validation_pos_rel].copy()
            if y_temp is None:
                y_train = None
                y_validation = None
            else:
                y_train = y_temp.iloc[train_pos_rel].copy()
                y_validation = y_temp.iloc[validation_pos_rel].copy()
        else:
            train_mask, validation_mask, test_mask = self._resolver_mascaras_grupos_ordenados(group_data, config)
            X_train = X.loc[train_mask].copy()
            X_validation = X.loc[validation_mask].copy() if validation_mask is not None else None
            X_test = X.loc[test_mask].copy()
            groups_train = group_data.loc[train_mask].copy()
            groups_validation = None if validation_mask is None else group_data.loc[validation_mask].copy()
            groups_test = group_data.loc[test_mask].copy()
            if y is None:
                y_train = None
                y_validation = None
                y_test = None
            else:
                y_train = y.loc[train_mask].copy()
                y_validation = None if validation_mask is None else y.loc[validation_mask].copy()
                y_test = y.loc[test_mask].copy()

        return self._construir_segmentos(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
            groups_train=groups_train,
            groups_validation=groups_validation,
            groups_test=groups_test,
        )

    def _resolver_mascaras_grupos_ordenados(
        self,
        group_data: pd.Series,
        config: BahamutSplitConfig,
    ) -> Tuple[pd.Series, Optional[pd.Series], pd.Series]:
        ordered_groups = pd.Index(pd.unique(group_data))
        total_groups = len(ordered_groups)
        n_test_groups = max(1, int(round(total_groups * config.test_size)))
        n_validation_groups = 0 if config.validation_size is None else max(1, int(round(total_groups * config.validation_size)))
        n_train_groups = total_groups - n_test_groups - n_validation_groups

        if n_train_groups < 1:
            raise ValueError("No quedan grupos suficientes para construir el bloque train.")

        train_groups = set(ordered_groups[:n_train_groups])
        validation_groups = set(ordered_groups[n_train_groups:n_train_groups + n_validation_groups])
        test_groups = set(ordered_groups[n_train_groups + n_validation_groups:])

        train_mask = group_data.isin(train_groups)
        validation_mask = None if not validation_groups else group_data.isin(validation_groups)
        test_mask = group_data.isin(test_groups)
        return train_mask, validation_mask, test_mask

    def _validar_estado_minimo(self) -> None:
        if self.problema is None:
            raise ValueError("Debes definir primero el tipo de problema con definir_problema().")

        if self.problema != "clusterizacion" and not self.target_cols:
            raise ValueError(
                "Debes definir al menos una variable objetivo con definir_objetivo() para problemas supervisados."
            )

        if self.problema == "series_temporales" and self.shuffle:
            raise ValueError(
                "Para series_temporales configura shuffle=False antes de ejecutar el split."
            )

        if self.problema == "series_temporales" and self.group_cols:
            raise ValueError(
                "La segmentacion temporal no se combina con group split. Usa time_col o cambia el tipo de problema."
            )

        if self.problema == "series_temporales" and self.time_col is None and not self.asumir_orden_actual:
            raise ValueError(
                "Una segmentacion temporal necesita saber el orden cronologico. "
                "Llama a definir_orden_temporal('mi_columna_de_fecha') o, si el DataFrame "
                "ya viene ordenado y quieres cortar por el orden de filas, pidelo de forma "
                "explicita con definir_orden_temporal(None, asumir_orden_actual=True)."
            )

        if self.stratify_mode == "columnas" and not self.stratify_cols:
            raise ValueError("Si usas modo='columnas', debes indicar stratify_cols validas.")

        if self.stratify_mode == "target" and not self.target_cols:
            raise ValueError("Si usas modo='target', primero debes definir target_cols.")

        if self.stratify_mode in self.MODOS_ESTRATIFICACION_EXPLICITOS and not self.shuffle:
            raise ValueError("No se puede estratificar con shuffle=False en train_test_split.")

        if self.group_cols and self.stratify_mode in self.MODOS_ESTRATIFICACION_EXPLICITOS:
            raise ValueError(
                "Bahamut no combina group split y estratificacion en esta version. Usa uno u otro."
            )

        target_vs_features = sorted(set(self.target_cols) & set(self.feature_cols))
        if target_vs_features:
            raise ValueError(f"No incluyas target_cols dentro de feature_cols: {target_vs_features}")

        target_vs_excludes = sorted(set(self.target_cols) & set(self.exclude_cols))
        if target_vs_excludes:
            raise ValueError(f"No incluyas target_cols dentro de exclude_cols: {target_vs_excludes}")

        target_vs_stratify = sorted(set(self.target_cols) & set(self.stratify_cols))
        if target_vs_stratify:
            raise ValueError(
                "No uses target_cols como stratify_cols. Usa modo='target' si quieres estratificar por objetivo."
            )

        if self.problema == "regresion" and self.stratify_mode in {"auto", "target"}:
            if len(self.target_cols) != 1:
                raise ValueError(
                    "La estratificacion automatica de regresion requiere una sola variable objetivo."
                )

        if self.group_cols:
            group_series = (
                self._colapsar_columnas_en_serie(self.df, list(self.group_cols))
                .astype("string")
                .fillna("__bahamut_nan_group__")
                .astype(str)
            )
            min_groups = 3 if self.validation_size is not None else 2
            if group_series.nunique(dropna=False) < min_groups:
                raise ValueError(
                    f"Necesitas al menos {min_groups} grupos distintos para la segmentacion por grupos actual."
                )

        _ = self._resolver_feature_cols()
        _ = self._resolver_train_size()


Bahamut_split = BahamutSplit


def demo_bahamut_split() -> Dict[str, Any]:
    """Pequena demo manual para revisar la API ampliada sin acoplarla a un notebook."""
    demo_df = pd.DataFrame(
        {
            "cliente_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "edad": [18, 19, 20, 21, 22, 23, 24, 25],
            "ingreso": [800, 900, 1100, 1300, 1400, 1600, 1800, 2000],
            "ciudad": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "compra": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )

    split_tool = (
        BahamutSplit(demo_df)
        .definir_problema("clasificacion")
        .definir_objetivo("compra")
        .definir_grupos("cliente_id")
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, validation_size=0.25, shuffle=True, random_state=123)
    )

    segmentos = split_tool.ejecutar_segmentacion()
    return {
        "variables": split_tool.variables_disponibles_para_split(),
        "resumen": split_tool.resumen_configuracion(),
        "diagnostico": split_tool.diagnostico_split(),
        "segmentos": split_tool.resumen_segmentos(segmentos),
    }
