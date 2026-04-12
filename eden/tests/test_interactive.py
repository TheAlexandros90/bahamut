import json

import pandas as pd

from eden import EdenWorkbench


def build_interactive_fixture() -> tuple[dict[str, object], pd.DataFrame]:
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=8, freq="D"),
            "store_id": ["A"] * 4 + ["B"] * 4,
            "trend": [1, 2, 3, 4, 1, 2, 3, 4],
            "promo": [0, 1, 0, 1, 0, 1, 0, 1],
            "y_sales": [10, 12, 11, 14, 9, 11, 10, 13],
            "split": ["train", "train", "valid", "test", "train", "train", "valid", "test"],
        }
    )
    train_idx = df.index[df["split"] == "train"].tolist()
    valid_idx = df.index[df["split"] == "valid"].tolist()
    test_idx = df.index[df["split"] == "test"].tolist()
    bahamut_segments = {
        "X_train": df.loc[train_idx, ["ds", "store_id", "trend", "promo"]].copy(),
        "X_validation": df.loc[valid_idx, ["ds", "store_id", "trend", "promo"]].copy(),
        "X_test": df.loc[test_idx, ["ds", "store_id", "trend", "promo"]].copy(),
        "y_train": df.loc[train_idx, ["y_sales"]].copy(),
        "y_validation": df.loc[valid_idx, ["y_sales"]].copy(),
        "y_test": df.loc[test_idx, ["y_sales"]].copy(),
        "indices_train": train_idx,
        "indices_validation": valid_idx,
        "indices_test": test_idx,
    }
    namespace = {
        "df_demo": df,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "bahamut_segmentos": bahamut_segments,
    }
    return namespace, df


def test_discover_sources_detects_dataframes_bahamut_and_indexers():
    namespace, _ = build_interactive_fixture()
    workbench = EdenWorkbench(namespace=namespace)
    sources = workbench.discover_sources()

    assert "df_demo" in sources["dataframes"]
    assert "bahamut_segmentos" in sources["bahamut_segments"]
    assert {"train_idx", "valid_idx", "test_idx"}.issubset(sources["indexers"])


def test_materialize_from_variables_supports_split_column():
    namespace, _ = build_interactive_fixture()
    workbench = EdenWorkbench(namespace=namespace)
    result = workbench.materialize_from_variables(
        "df_demo",
        timestamp_col="ds",
        target_cols=["y_sales"],
        feature_cols=["trend", "promo"],
        entity_id_col="store_id",
        split_col="split",
    )

    assert result.spec.series_structure == "panel"
    assert result.splits.available_splits() == ["train", "valid", "test"]
    assert list(result.summary()["split"]) == ["train", "valid", "test"]


def test_materialize_from_bahamut_returns_ready_spec_and_partitions():
    namespace, _ = build_interactive_fixture()
    workbench = EdenWorkbench(namespace=namespace)
    result = workbench.materialize_from_bahamut(
        "bahamut_segmentos",
        timestamp_col="ds",
        target_cols=["y_sales"],
        feature_cols=["trend", "promo"],
        entity_id_col="store_id",
    )

    assert result.source_mode == "bahamut"
    assert result.spec.target_cols == ["y_sales"]
    assert "test" in result.partitions


def test_materialize_from_files_supports_csv_and_json_indices(tmp_path):
    namespace, df = build_interactive_fixture()
    workbench = EdenWorkbench(namespace=namespace)

    df_path = tmp_path / "dataset.csv"
    train_path = tmp_path / "train_idx.json"
    valid_path = tmp_path / "valid_idx.json"
    test_path = tmp_path / "test_idx.json"

    df.to_csv(df_path, index=False)
    train_path.write_text(json.dumps([0, 1, 4, 5]), encoding="utf-8")
    valid_path.write_text(json.dumps([2, 6]), encoding="utf-8")
    test_path.write_text(json.dumps([3, 7]), encoding="utf-8")

    result = workbench.materialize_from_files(
        df_path,
        timestamp_col="ds",
        target_cols=["y_sales"],
        feature_cols=["trend", "promo"],
        entity_id_col="store_id",
        train_idx_source=train_path,
        valid_idx_source=valid_path,
        test_idx_source=test_path,
    )

    assert result.splits.available_splits() == ["train", "valid", "test"]
    assert len(result.partitions["test"]) == 2