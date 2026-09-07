"""Regresiones de los defectos detectados en la auditoria de 2026."""

from __future__ import annotations

import pandas as pd
import pytest

from bahamut import BahamutSplit


def build_frame(shuffled: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "fecha": pd.date_range("2024-01-01", periods=24, freq="D"),
            "cliente": [index // 3 for index in range(24)],
            "x": range(24),
            "y": [index % 2 for index in range(24)],
        }
    )
    if shuffled:
        frame = frame.sample(frac=1, random_state=0)
    return frame


# --------------------------------------------------------------------------
# Fuga temporal silenciosa
# --------------------------------------------------------------------------

def test_series_temporales_sin_time_col_falla_en_vez_de_filtrar_el_futuro():
    splitter = (
        BahamutSplit(build_frame(shuffled=True))
        .definir_problema("series_temporales")
        .definir_objetivo("y")
        .definir_predictoras(exclude_cols=["fecha", "cliente"])
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, shuffle=False)
    )

    with pytest.raises(ValueError, match="orden cronologico"):
        splitter.ejecutar_segmentacion()


def test_series_temporales_permite_el_orden_de_filas_si_se_pide_explicitamente():
    frame = build_frame()
    splitter = (
        BahamutSplit(frame)
        .definir_problema("series_temporales")
        .definir_objetivo("y")
        .definir_predictoras(exclude_cols=["fecha", "cliente"])
        .definir_orden_temporal(None, asumir_orden_actual=True)
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, shuffle=False)
    )

    segmentos = splitter.ejecutar_segmentacion()
    diagnostico = splitter.diagnostico_split()

    assert frame.loc[segmentos["indices_train"], "fecha"].max() < frame.loc[segmentos["indices_test"], "fecha"].min()
    assert "orden actual de las filas" in diagnostico["orden_temporal"]
    assert "orden actual de las filas" in splitter.parametros_para_split_manual()["nota_temporal"]


def test_series_temporales_con_time_col_ordena_aunque_el_df_venga_desordenado():
    frame = build_frame(shuffled=True)
    splitter = (
        BahamutSplit(frame)
        .definir_problema("series_temporales")
        .definir_objetivo("y")
        .definir_predictoras(exclude_cols=["fecha", "cliente"])
        .definir_orden_temporal("fecha", ascending=True)
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, shuffle=False)
    )

    segmentos = splitter.ejecutar_segmentacion()

    assert frame.loc[segmentos["indices_train"], "fecha"].max() < frame.loc[segmentos["indices_test"], "fecha"].min()
    assert "ordenado por 'fecha'" in splitter.diagnostico_split()["orden_temporal"]


# --------------------------------------------------------------------------
# "auto" es una preferencia, no un requisito
# --------------------------------------------------------------------------

def test_estratificacion_auto_no_bloquea_shuffle_false():
    splitter = (
        BahamutSplit(build_frame())
        .definir_problema("clasificacion")
        .definir_objetivo("y")
        .configurar_split(test_size=0.25, shuffle=False)
    )

    diagnostico = splitter.diagnostico_split()

    assert diagnostico["estratificacion_activada"] is False
    assert diagnostico["modo_estratificacion_efectivo"] == "desactivada: shuffle=False"


def test_estratificacion_auto_no_bloquea_group_split():
    splitter = (
        BahamutSplit(build_frame())
        .definir_problema("clasificacion")
        .definir_objetivo("y")
        .definir_grupos("cliente")
        .configurar_split(test_size=0.25, shuffle=True, random_state=3)
    )

    segmentos = splitter.ejecutar_segmentacion()
    diagnostico = splitter.diagnostico_split()

    assert diagnostico["modo_estratificacion_efectivo"] == "desactivada: incompatible con group split"
    grupos_train = set(segmentos["groups_train"])
    grupos_test = set(segmentos["groups_test"])
    assert grupos_train.isdisjoint(grupos_test)


def test_estratificacion_explicita_sigue_siendo_incompatible_con_shuffle_false():
    with pytest.raises(ValueError, match="No se puede estratificar"):
        (
            BahamutSplit(build_frame())
            .definir_problema("clasificacion")
            .definir_objetivo("y")
            .definir_estratificacion(modo="target")
            .configurar_split(test_size=0.25, shuffle=False)
        )


# --------------------------------------------------------------------------
# La validacion no debe dejar el objeto a medio configurar
# --------------------------------------------------------------------------

def test_definir_estratificacion_invalida_no_muta_el_estado():
    splitter = BahamutSplit(build_frame()).definir_problema("clasificacion").definir_objetivo("y")
    estado_previo = (splitter.stratify_mode, splitter.regression_bins, list(splitter.stratify_cols))

    with pytest.raises(ValueError, match="Solo puedes pasar stratify_cols"):
        splitter.definir_estratificacion(["x"], modo="target")

    assert (splitter.stratify_mode, splitter.regression_bins, list(splitter.stratify_cols)) == estado_previo


def test_definir_estratificacion_con_columna_inexistente_no_muta_el_estado():
    splitter = BahamutSplit(build_frame()).definir_problema("clasificacion").definir_objetivo("y")

    with pytest.raises(ValueError, match="Columnas no encontradas"):
        splitter.definir_estratificacion(["no_existe"], modo="columnas")

    assert splitter.stratify_mode == "auto"
    assert splitter.stratify_cols == []


# --------------------------------------------------------------------------
# El nucleo no debe depender del stack de notebook
# --------------------------------------------------------------------------

def test_el_nucleo_no_importa_ipywidgets():
    import bahamut.split as split_module

    fuente = split_module.__file__
    with open(fuente, encoding="utf-8") as handle:
        contenido = handle.read()

    assert "import ipywidgets" not in contenido
    assert "from IPython" not in contenido


def test_la_tabla_interactiva_sigue_colgando_de_la_clase():
    assert callable(getattr(BahamutSplit, "tabla_interactiva", None))
    assert callable(getattr(BahamutSplit, "interactive_table", None))


def test_tabla_interactiva_se_construye_sin_romper():
    widgets = pytest.importorskip("ipywidgets")

    splitter = (
        BahamutSplit(build_frame())
        .definir_problema("clasificacion")
        .definir_objetivo("y")
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, shuffle=True, random_state=1)
    )

    panel = splitter.tabla_interactiva(preview_rows=4)

    assert isinstance(panel, widgets.Widget)
