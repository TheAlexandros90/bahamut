"""Capa interactiva de Bahamut (tabla de segmentacion para notebook).

Se mantiene fuera de `split.py` para que el nucleo analitico no dependa de
ipywidgets ni de IPython. `bahamut/__init__.py` engancha estos metodos a
`BahamutSplit` al importar el paquete, igual que hace Fenrir con su clase.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ._notebook import (
    clear_output,
    display,
    publish_notebook_bindings,
    require_widgets,
    round_frame,
    widgets,
)
from .split import BahamutSplit


def _tabla_interactiva(self, preview_rows: int = 6):
    """Tabla interactiva para definir y ejecutar la segmentacion."""
    require_widgets("bahamut")

    columnas = self.columnas_disponibles()
    state: Dict[str, Any] = {"model": self, "segmentos": None}

    problem_widget = widgets.Dropdown(
        options=[(value.capitalize(), value) for value in sorted(self.PROBLEMAS_VALIDOS)],
        value=self.problema or "clasificacion",
        description="Problema",
        layout=widgets.Layout(width="280px"),
    )
    target_widget = widgets.SelectMultiple(
        options=columnas,
        value=tuple(self.target_cols),
        description="Target",
        rows=min(8, max(4, len(columnas))),
        layout=widgets.Layout(width="260px", height="190px"),
    )
    use_features_widget = widgets.Checkbox(value=bool(self.feature_cols), description="Usar features")
    features_widget = widgets.SelectMultiple(
        options=columnas,
        value=tuple(self.feature_cols),
        description="Features",
        rows=min(8, max(4, len(columnas))),
        layout=widgets.Layout(width="260px", height="190px"),
    )
    exclude_widget = widgets.SelectMultiple(
        options=columnas,
        value=tuple(self.exclude_cols),
        description="Excluir",
        rows=min(8, max(4, len(columnas))),
        layout=widgets.Layout(width="260px", height="190px"),
    )
    stratify_mode_widget = widgets.Dropdown(
        options=[
            ("Auto", "auto"),
            ("Target", "target"),
            ("Columnas", "columnas"),
            ("Ninguna", "ninguna"),
        ],
        value=self.stratify_mode,
        description="Estrat.",
        layout=widgets.Layout(width="260px"),
    )
    stratify_cols_widget = widgets.SelectMultiple(
        options=columnas,
        value=tuple(self.stratify_cols),
        description="Strat cols",
        rows=min(6, max(4, len(columnas))),
        layout=widgets.Layout(width="260px", height="160px"),
    )
    group_cols_widget = widgets.SelectMultiple(
        options=columnas,
        value=tuple(self.group_cols),
        description="Grupos",
        rows=min(6, max(4, len(columnas))),
        layout=widgets.Layout(width="260px", height="160px"),
    )
    test_size_widget = widgets.FloatSlider(
        value=float(self.test_size),
        min=0.05,
        max=0.45,
        step=0.05,
        readout_format=".2f",
        description="Test",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )
    use_validation_widget = widgets.Checkbox(
        value=self.validation_size is not None,
        description="Con valid",
    )
    validation_size_widget = widgets.FloatSlider(
        value=0.2 if self.validation_size is None else float(self.validation_size),
        min=0.05,
        max=0.45,
        step=0.05,
        readout_format=".2f",
        description="Valid",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )
    shuffle_widget = widgets.Checkbox(value=bool(self.shuffle), description="Shuffle")
    random_state_widget = widgets.IntText(
        value=42 if self.random_state is None else int(self.random_state),
        description="Seed",
        layout=widgets.Layout(width="200px"),
    )
    regression_bins_widget = widgets.IntSlider(
        value=5 if self.regression_bins is None else int(self.regression_bins),
        min=2,
        max=10,
        step=1,
        description="Bins reg",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )
    time_col_widget = widgets.Dropdown(
        options=[("Sin col temporal: usar orden de filas", "__none__")] + [(col, col) for col in columnas],
        value="__none__" if self.time_col is None else self.time_col,
        description="Time col",
        layout=widgets.Layout(width="280px"),
    )
    time_order_widget = widgets.Checkbox(value=self.time_ascending, description="Asc temporal")
    view_widget = widgets.Dropdown(
        options=[
            ("Resumen segmentos", "summary"),
            ("Balance target", "balance"),
            ("Preview segmento", "preview"),
            ("Parametros manuales", "manual"),
            ("Variables split", "variables"),
        ],
        value="summary",
        description="Vista",
        layout=widgets.Layout(width="280px"),
    )
    preview_segment_widget = widgets.Dropdown(
        options=[("Train", "train"), ("Test", "test")],
        value="train",
        description="Preview",
        layout=widgets.Layout(width="260px"),
    )
    preview_rows_widget = widgets.IntSlider(
        value=max(3, int(preview_rows)),
        min=3,
        max=20,
        step=1,
        description="Filas",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )
    round_widget = widgets.Dropdown(
        options=[("Sin redondeo", None)] + [(str(value), value) for value in range(0, 7)],
        value=4,
        description="Round",
        layout=widgets.Layout(width="220px"),
    )
    refresh_button = widgets.Button(
        description="Actualizar split",
        button_style="primary",
        icon="refresh",
    )

    status_output = widgets.Output()
    summary_output = widgets.Output()
    table_output = widgets.Output()

    def _sync_options(*_):
        selected_targets = set(target_widget.value)
        feature_options = [col for col in columnas if col not in selected_targets]
        current_features = tuple(col for col in features_widget.value if col in feature_options)
        current_excludes = tuple(col for col in exclude_widget.value if col in feature_options)
        current_stratify = tuple(col for col in stratify_cols_widget.value if col in feature_options)
        current_groups = tuple(col for col in group_cols_widget.value if col in columnas)

        features_widget.options = feature_options
        exclude_widget.options = feature_options
        stratify_cols_widget.options = feature_options
        group_cols_widget.options = columnas
        features_widget.value = current_features
        exclude_widget.value = current_excludes
        stratify_cols_widget.value = current_stratify
        group_cols_widget.value = current_groups

        validation_size_widget.disabled = not use_validation_widget.value
        regression_bins_widget.disabled = problem_widget.value != "regresion"
        time_order_widget.disabled = time_col_widget.value == "__none__"

        preview_options = [("Train", "train"), ("Test", "test")]
        if use_validation_widget.value:
            preview_options.insert(1, ("Validation", "validation"))
        preview_segment_widget.options = preview_options
        if preview_segment_widget.value not in dict(preview_options).values():
            preview_segment_widget.value = preview_options[0][1]

        if problem_widget.value == "series_temporales":
            shuffle_widget.value = False
            shuffle_widget.disabled = True
            if stratify_mode_widget.value != "ninguna":
                stratify_mode_widget.value = "ninguna"
            if group_cols_widget.value:
                group_cols_widget.value = ()
            group_cols_widget.disabled = True
        else:
            shuffle_widget.disabled = False
            group_cols_widget.disabled = False

        group_split_active = bool(group_cols_widget.value)
        if group_split_active and stratify_mode_widget.value != "ninguna":
            stratify_mode_widget.value = "ninguna"

        stratify_mode_widget.disabled = problem_widget.value == "series_temporales" or group_split_active
        stratify_cols_widget.disabled = stratify_mode_widget.value != "columnas" or group_split_active

    def _build_model() -> "BahamutSplit":
        selected_targets = list(target_widget.value)
        selected_features = list(features_widget.value) if use_features_widget.value else None
        selected_excludes = list(exclude_widget.value)
        selected_groups = list(group_cols_widget.value)
        validation_size = float(validation_size_widget.value) if use_validation_widget.value else None
        time_col = None if time_col_widget.value == "__none__" else time_col_widget.value
        stratify_cols = list(stratify_cols_widget.value) if stratify_mode_widget.value == "columnas" else None
        random_state = int(random_state_widget.value) if shuffle_widget.value else None

        model = (
            BahamutSplit(self.df.copy())
            .definir_problema(problem_widget.value)
            .definir_objetivo(selected_targets or None)
            .definir_predictoras(selected_features, selected_excludes)
            .definir_grupos(selected_groups or None)
            .definir_orden_temporal(
                time_col,
                ascending=time_order_widget.value,
                # Elegir "Sin col temporal" en la tabla es el consentimiento
                # explicito a cortar por el orden actual de las filas.
                asumir_orden_actual=time_col is None,
            )
            .definir_estratificacion(
                stratify_cols,
                modo=stratify_mode_widget.value,
                bins_regresion=int(regression_bins_widget.value),
            )
            .configurar_split(
                test_size=float(test_size_widget.value),
                validation_size=validation_size,
                shuffle=bool(shuffle_widget.value),
                random_state=random_state,
            )
        )
        return model

    def _render(_=None):
        with status_output:
            clear_output(wait=True)
            print("Actualizando tabla de split...")

        try:
            model = _build_model()
            segmentos = model.ejecutar_segmentacion()
            state["model"] = model
            state["segmentos"] = segmentos

            config_frame = pd.Series(model.resumen_configuracion(), name="valor").to_frame()
            diagnostic_frame = pd.Series(model.diagnostico_split(), name="valor").to_frame()
            config_frame = round_frame(config_frame, round_widget.value)
            diagnostic_frame = round_frame(diagnostic_frame, round_widget.value)

            if view_widget.value == "summary":
                frame = model.resumen_segmentos(segmentos)
            elif view_widget.value == "manual":
                frame = pd.Series(model.parametros_para_split_manual(), name="valor").to_frame()
            elif view_widget.value == "balance":
                try:
                    frame = model.balance_target_por_segmento(segmentos)
                except Exception as exc:
                    frame = pd.DataFrame({"info": [str(exc)]})
            elif view_widget.value == "variables":
                frame = model.tabla_variables_split()
            else:
                frame = model.preview_segmento(
                    segmentos,
                    segmento=preview_segment_widget.value,
                    rows=int(preview_rows_widget.value),
                )

            frame = round_frame(frame, round_widget.value)

            with summary_output:
                clear_output(wait=True)
                print("Configuracion activa")
                display(config_frame)
                print("Diagnostico")
                display(diagnostic_frame)

            with table_output:
                clear_output(wait=True)
                display(frame)

            publish_notebook_bindings(
                bahamut_split=model,
                bahamut_segmentos=segmentos,
                bahamut_split_view=frame,
            )

            with status_output:
                clear_output(wait=True)
                print("Tabla de split lista")
        except Exception as exc:
            with summary_output:
                clear_output(wait=True)
            with table_output:
                clear_output(wait=True)
            with status_output:
                clear_output(wait=True)
                print(f"No se pudo construir la tabla de split: {exc}")

    target_widget.observe(_sync_options, names="value")
    use_validation_widget.observe(_sync_options, names="value")
    stratify_mode_widget.observe(_sync_options, names="value")
    problem_widget.observe(_sync_options, names="value")
    time_col_widget.observe(_sync_options, names="value")
    group_cols_widget.observe(_sync_options, names="value")
    refresh_button.on_click(_render)

    _sync_options()
    _render()

    controls = widgets.VBox(
        [
            widgets.HTML("<b>Tabla interactiva de BahamutSplit</b>"),
            widgets.HTML(
                "<i>Configura train/test o train/validation/test, define variables, grupos, estratificacion y revisa el diagnostico antes de ejecutar.</i>"
            ),
            problem_widget,
            target_widget,
            use_features_widget,
            widgets.HBox([features_widget, exclude_widget]),
            stratify_mode_widget,
            widgets.HBox([stratify_cols_widget, group_cols_widget]),
            widgets.HBox([test_size_widget, validation_size_widget]),
            widgets.HBox([use_validation_widget, shuffle_widget]),
            widgets.HBox([random_state_widget, regression_bins_widget]),
            widgets.HBox([time_col_widget, time_order_widget]),
            view_widget,
            widgets.HBox([preview_segment_widget, preview_rows_widget]),
            round_widget,
            refresh_button,
            status_output,
        ]
    )
    content = widgets.VBox([summary_output, table_output], layout=widgets.Layout(width="100%"))
    return widgets.HBox([controls, content], layout=widgets.Layout(align_items="flex-start"))


def attach_interactive_api(cls=BahamutSplit):
    """Engancha la tabla interactiva a la clase de segmentacion."""

    if getattr(cls, "_interactive_api_attached", False):
        return cls

    cls.tabla_interactiva = _tabla_interactiva
    cls.interactive_table = _tabla_interactiva
    cls._interactive_api_attached = True
    return cls


attach_interactive_api(BahamutSplit)
