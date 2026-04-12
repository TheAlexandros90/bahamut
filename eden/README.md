# Eden

Eden es una biblioteca orientada exclusivamente a forecasting temporal.

Principios del paquete:

- Bahamut decide los cortes `train`/`valid`/`test` cuando el flujo de trabajo ya viene segmentado.
- Eden se encarga de validacion temporal, entrenamiento, prediccion, metricas, intervalos y backtesting.
- Eden puede consumir bundles de Bahamut, `DataFrame` con splits externos o una configuracion interactiva desde notebook.

## Instalacion

Si clonas el repo padre desde GitHub, Eden vive dentro de la carpeta `eden/`:

```text
<nombre-del-repo>/
`-- eden/
    |-- pyproject.toml
    |-- README.md
    |-- examples/
    |-- src/
    |   `-- eden/
    `-- tests/
```

Instalacion base desde la raiz del repo padre:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e "./eden"
```

Si tambien quieres Bahamut en el mismo entorno:

```bash
pip install -e ".[notebook]"
pip install -e "./eden[notebook]"
```

Si entras directamente en la carpeta `eden/`, los comandos equivalentes son estos:

```bash
pip install -e ".[notebook]"
pip install -e ".[dev,notebook]"
pip install -e ".[notebook,prophet]"
```

`venv` no es obligatorio. Solo se recomienda para no mezclar dependencias del ejemplo con el Python global del equipo. Si alguien solo quiere probar el CSV incluido o abrir el notebook una vez, puede omitirlo y usar su entorno activo.

Instalacion base:

```bash
pip install -e "./eden"
```

Con soporte interactivo para notebook:

```bash
pip install -e "./eden[notebook]"
```

Con extras de desarrollo:

```bash
pip install -e "./eden[dev,notebook]"
```

Con Prophet:

```bash
pip install -e "./eden[notebook,prophet]"
```

## Uso rapido

```python
from sklearn.linear_model import LinearRegression

from eden import Eden, EdenSpec, EdenSplits

spec = EdenSpec(
    timestamp_col="ds",
    target_cols=["y"],
    feature_cols=["trend", "promo"],
    metrics=["mae", "rmse", "wape"],
)

splits = EdenSplits(train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

model = Eden(spec=spec, model=LinearRegression())
model.fit(df, splits)
predicciones = model.predict_split(df, splits, split_name="test")
```

## Ejemplo real incluido

El repo incluye un dataset real y listo para cargar desde notebook o desde la tabla interactiva:

- `examples/shampoo_sales_monthly.csv`

El archivo ya trae:

- la serie temporal `sales`;
- covariables futuras deterministas `trend`, `month_sin`, `month_cos`;
- una columna `split` con los cortes `train` / `valid` / `test`.

Ejemplo directo:

```python
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from eden import Eden, EdenSpec, EdenSplits

example_path = Path("eden/examples/shampoo_sales_monthly.csv")
df = pd.read_csv(example_path, parse_dates=["ds"])

splits = EdenSplits(
    train_idx=df.index[df["split"] == "train"].tolist(),
    valid_idx=df.index[df["split"] == "valid"].tolist(),
    test_idx=df.index[df["split"] == "test"].tolist(),
)

spec = EdenSpec(
    timestamp_col="ds",
    target_cols=["sales"],
    feature_cols=["trend", "month_sin", "month_cos"],
    metrics=["mae", "rmse", "wape"],
    interval_coverage=0.9,
    calibration_split="valid",
    known_future_feature_cols=["trend", "month_sin", "month_cos"],
    enforce_future_covariates=True,
    expected_frequency="MS",
)

model = Eden(spec=spec, model=LinearRegression())
model.fit(df, splits)
predicciones = model.predict_with_metrics(df.loc[splits.test_idx].copy())
```

## Interactividad

La clase `EdenWorkbench` expone una tabla interactiva pensada para notebook que permite:

- consumir un bundle Bahamut ya creado;
- seleccionar un `DataFrame` y variables `train_idx`/`valid_idx`/`test_idx` desde el notebook;
- cargar archivos locales y construir la configuracion temporal sin escribir codigo adicional.
- elegir una receta de modelado lista para usar: lineal, cuantiles lineales o Prophet;
- usar un blueprint de modelo o un adapter ya definidos en el notebook;
- entrenar y predecir desde la propia tabla, publicando `eden_model`, `eden_run_result`, `eden_predictions`, `eden_metrics` y `eden_performance_report`.

La tabla publica en el namespace del notebook objetos listos para usar como `eden_spec`, `eden_splits`, `eden_source_df` y `eden_partitions`.

Ejemplo rapido en notebook:

```python
from IPython.display import display

from eden import EdenWorkbench

workbench = EdenWorkbench(namespace=globals())
display(workbench.tabla_interactiva(preview_rows=5))

# Despues de preparar la configuracion desde la tabla:
run_result = workbench.train_and_predict(runtime="linear", prediction_split="test")
display(run_result.selected_predictions)
display(run_result.performance_report)
```

## Como funciona la tabla

Lectura rapida de izquierda a derecha:

- `Fuente` decide de donde sale el dataset: un bundle de Bahamut ya existente, variables del notebook o archivos locales.
- `Bahamut` usa un bundle con `X_*`, `y_*` e indices ya generados. `df original` es opcional y solo hace falta si quieres reconstruir columnas que no viajan en el bundle.
- `DataFrame` usa una variable `pandas.DataFrame` del notebook. En `Split` puedes elegir entre variables `train_idx` / `valid_idx` / `test_idx` o una `Col split` dentro del propio dataframe.
- `train_idx`, `valid_idx` y `test_idx` no son datasets nuevos: son listas de indices o etiquetas de fila del mismo `DataFrame` seleccionado. Sirven para decirle a Eden que filas pertenecen a train, valid y test.
- Si eliges `Fuente = Variables del notebook`, solo debes seleccionar variables `*_idx` creadas a partir de ese mismo dataframe. Ejemplo: si el `DataFrame` es `panel_df`, las variables correctas son las que se calcularon sobre `panel_df`, no indices de otro dataframe distinto.
- `train_idx` es obligatorio. `valid_idx` y `test_idx` pueden faltar, pero entonces algunas metricas, intervalos o vistas quedaran limitadas.
- Si no quieres pensar en indices manuales, usa `Col split`: suele ser la opcion mas clara para compartir el notebook con otra persona.
- `Archivos locales` acepta una ruta o una subida de archivo para el dataframe. Los splits pueden venir de archivos de indices o de una columna `split` dentro del CSV o Excel.
- `Timestamp`, `Entidad`, `Target` y `Features` definen el `EdenSpec` que se va a construir.
- `Preparar Eden` no entrena nada. Solo valida la configuracion, construye `eden_spec`, `eden_splits`, `eden_source_df`, `eden_partitions`, `eden_workbench` y `eden_workbench_result`, y actualiza las vistas de resumen y preview.
- `Modelo` elige la receta de entrenamiento: lineal sklearn, cuantiles lineales, Prophet, un modelo del notebook o un adapter del notebook.
- `Modelo ns` y `Adapter ns` solo importan cuando el runtime elegido es `Modelo notebook` o `Adapter notebook`.
- `Predecir`, `Cobertura` y `Reporte con train` controlan la ejecucion del modelo y el tipo de reporte generado.
- `Entrenar y predecir` ajusta el modelo con la configuracion resuelta y publica `eden_model`, `eden_run_result`, `eden_predictions`, `eden_metrics` y `eden_performance_report`.
- `Vista` cambia lo que aparece a la derecha: resumen, preview, config, predicciones, reporte o diagnostico.
- `Preview` y `Filas` solo cambian la visualizacion. No afectan al entrenamiento.

Receta rapida para alguien que llega nuevo al repo y quiere probar el ejemplo real incluido:

- Selecciona `Fuente = Archivos locales`.
- En `Archivo df` usa `eden/examples/shampoo_sales_monthly.csv` o sube ese mismo archivo.
- En `Split` elige `Columna de split`.
- En `Col split` selecciona `split`.
- En `Timestamp` selecciona `ds`.
- En `Target` selecciona `sales`.
- En `Features` deja `trend`, `month_sin` y `month_cos`.
- Pulsa `Preparar Eden` para que el notebook publique `eden_spec` y `eden_splits`.
- Elige `Modelo = Lineal sklearn` o `Prophet` y pulsa `Entrenar y predecir`.
- Cambia `Vista` a `Predicciones` o `Reporte` para ver la salida final.

## Portabilidad a otro PC

Para que este notebook funcione en otro ordenador sin depender de rutas tipo `c:\Users\...`, la regla es esta:

- primero intenta trabajar con `pip install -e "./eden[notebook]"` desde la raiz del repo;
- si el notebook esta al lado del repo, puede detectar `eden/src` automaticamente como fallback local;
- evita hardcodear rutas absolutas en notebooks futuros, tambien en proyectos como Fenrir.

La idea reusable es sencilla: imports desde paquete instalable y, solo como respaldo, deteccion relativa del repo. Esa es la forma de que el cambio de ordenador no rompa la carga.