import pandas as pd
import pytest

from bahamut import BahamutSplit


def build_frame() -> pd.DataFrame:
    rows = []
    for group_id in range(1, 9):
        for offset in range(3):
            idx = (group_id - 1) * 3 + offset
            rows.append(
                {
                    "fecha": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                    "cliente_id": group_id,
                    "edad": 20 + idx,
                    "ingreso": 1000 + idx * 75,
                    "ciudad": ["A", "B", "C", "D"][idx % 4],
                    "compra": idx % 2,
                    "objetivo_reg": 10.0 + idx * 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_variables_disponibles_para_split_cubre_roles_principales():
    df = build_frame()
    splitter = (
        BahamutSplit(df)
        .definir_objetivo("compra")
        .definir_predictoras(exclude_cols=["fecha"])
        .definir_grupos("cliente_id")
    )

    variables = splitter.variables_disponibles_para_split()

    assert "columnas_candidatas_target" in variables
    assert "columnas_candidatas_group" in variables
    assert "feature_cols_resueltas" in variables
    assert "compra" in variables["columnas_candidatas_target"]
    assert "cliente_id" in variables["group_cols_actuales"]
    assert "compra" not in variables["feature_cols_resueltas"]


def test_clasificacion_tripartita_con_estratificacion_por_target():
    df = build_frame()
    splitter = (
        BahamutSplit(df)
        .definir_problema("clasificacion")
        .definir_objetivo("compra")
        .definir_estratificacion(modo="target")
        .configurar_split(test_size=0.25, validation_size=0.25, shuffle=True, random_state=7)
    )

    segmentos = splitter.ejecutar_segmentacion()

    assert segmentos["X_validation"] is not None
    assert len(segmentos["X_train"]) + len(segmentos["X_validation"]) + len(segmentos["X_test"]) == len(df)
    assert set(segmentos["y_train"].unique()) == {0, 1}
    assert list(segmentos["resumen_segmentos"]["segmento"]) == ["train", "validation", "test"]


def test_clusterizacion_sin_target_devuelve_y_none():
    df = build_frame().drop(columns=["compra", "objetivo_reg"])
    splitter = (
        BahamutSplit(df)
        .definir_problema("clusterizacion")
        .definir_objetivo(None)
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, validation_size=0.25, shuffle=True, random_state=7)
    )

    segmentos = splitter.ejecutar_segmentacion()

    assert segmentos["y_train"] is None
    assert segmentos["y_validation"] is None
    assert segmentos["y_test"] is None


def test_series_temporales_respeta_el_orden():
    df = build_frame()
    splitter = (
        BahamutSplit(df)
        .definir_problema("series_temporales")
        .definir_objetivo("objetivo_reg")
        .definir_predictoras(exclude_cols=["fecha", "compra", "cliente_id"])
        .definir_orden_temporal("fecha", ascending=True)
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, validation_size=0.25, shuffle=False)
    )

    segmentos = splitter.ejecutar_segmentacion()

    fechas_train = df.loc[segmentos["indices_train"], "fecha"]
    fechas_validation = df.loc[segmentos["indices_validation"], "fecha"]
    fechas_test = df.loc[segmentos["indices_test"], "fecha"]

    assert fechas_train.max() < fechas_validation.min()
    assert fechas_validation.max() < fechas_test.min()


def test_group_split_no_mezcla_entidades_entre_segmentos():
    df = build_frame()
    splitter = (
        BahamutSplit(df)
        .definir_problema("clasificacion")
        .definir_objetivo("compra")
        .definir_grupos("cliente_id")
        .definir_estratificacion(modo="ninguna")
        .configurar_split(test_size=0.25, validation_size=0.25, shuffle=True, random_state=13)
    )

    segmentos = splitter.ejecutar_segmentacion()

    grupos_train = set(df.loc[segmentos["indices_train"], "cliente_id"])
    grupos_validation = set(df.loc[segmentos["indices_validation"], "cliente_id"])
    grupos_test = set(df.loc[segmentos["indices_test"], "cliente_id"])

    assert grupos_train.isdisjoint(grupos_validation)
    assert grupos_train.isdisjoint(grupos_test)
    assert grupos_validation.isdisjoint(grupos_test)
    assert "grupos_unicos" in segmentos["resumen_segmentos"].columns


def test_group_split_y_estratificacion_no_se_permiten_juntos():
    df = build_frame()
    with pytest.raises(ValueError, match="group split y estratificacion"):
        (
            BahamutSplit(df)
            .definir_problema("clasificacion")
            .definir_objetivo("compra")
            .definir_grupos("cliente_id")
            .definir_estratificacion(modo="target")
            .configurar_split(test_size=0.25, shuffle=True, random_state=3)
        )


def test_regresion_con_bins_aparece_en_diagnostico():
    df = build_frame()
    splitter = (
        BahamutSplit(df)
        .definir_problema("regresion")
        .definir_objetivo("objetivo_reg")
        .definir_predictoras(exclude_cols=["fecha", "compra"])
        .definir_estratificacion(modo="target", bins_regresion=4)
        .configurar_split(test_size=0.25, shuffle=True, random_state=11)
    )

    diagnostico = splitter.diagnostico_split()

    assert diagnostico["estratificacion_source"] == "target_bins"
    assert diagnostico["metodo_split"] == "train_test_split"