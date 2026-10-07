# Bahamut

Base inicial para utilidades de machine learning en Bahamut.

La primera pieza del proyecto es un motor de segmentacion que cubre los casos mas comunes de trabajo real:

- Split `train/test` clasico.
- Split `train/validation/test`.
- Estratificacion por target o por columnas arbitrarias.
- Estratificacion de regresion por cuantiles.
- Segmentacion temporal ordenada.
- Segmentacion por grupos para evitar leakage entre entidades.
- Inspeccion de todas las variables disponibles para configurar el split.
- Tabla interactiva opcional para notebook.

## Instalacion

Si vas a clonar el repositorio completo desde GitHub, la estructura relevante es esta:

```text
<nombre-del-repo>/
|-- pyproject.toml
|-- README.md
|-- src/
|   `-- bahamut/
|-- tests/
`-- eden/
    |-- pyproject.toml
    |-- README.md
    |-- examples/
    |-- src/
    |   `-- eden/
    `-- tests/
```

Instalacion recomendada para trabajar con Bahamut y Eden en cualquier ordenador:

```bash
git clone https://github.com/TheAlexandors90/<nombre-del-repo>.git
cd <nombre-del-repo>
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[notebook]"
pip install -e "./eden[notebook]"
```

Si solo quieres Bahamut:

```bash
pip install -e ".[notebook]"
```

Si solo quieres Eden:

```bash
pip install -e "./eden[notebook]"
```

Si quieres usar Prophet dentro de Eden:

```bash
pip install -e "./eden[notebook,prophet]"
```

`venv` es opcional. No se sube al repo, esta ignorado por `.gitignore` y no ensucia GitHub ni la documentacion. Solo crea un entorno aislado para no mezclar dependencias del proyecto con tu Python global. Si solo quieres probar el ejemplo una vez, puedes omitirlo y ejecutar los mismos `pip install -e ...` sobre tu Python activo, pero para desarrollo, tests o notebooks reproducibles es mejor mantenerlo.

```bash
pip install -e ".[dev]"
```

Para usar la vista interactiva en notebooks:

```bash
pip install -e ".[dev,notebook]"
```

## Uso rapido

```python
import pandas as pd

from bahamut import BahamutSplit

df = pd.DataFrame(
    {
        "cliente_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "edad": [21, 22, 35, 36, 48, 49, 29, 30],
        "ingreso": [1000, 1050, 1500, 1550, 2200, 2250, 1300, 1350],
        "compra": [0, 0, 1, 1, 0, 0, 1, 1],
    }
)

splitter = (
    BahamutSplit(df)
    .definir_problema("clasificacion")
    .definir_objetivo("compra")
    .definir_grupos("cliente_id")
    .definir_estratificacion(modo="ninguna")
    .configurar_split(test_size=0.25, validation_size=0.25, shuffle=True, random_state=7)
)

print(splitter.variables_disponibles_para_split())
segmentos = splitter.ejecutar_segmentacion()
print(segmentos["resumen_segmentos"])
```

## API principal

### Segmentacion temporal

Para `series_temporales`, indica la columna que define el orden cronologico:

```python
splitter = (
    BahamutSplit(df)
    .definir_problema("series_temporales")
    .definir_objetivo("ventas")
    .definir_orden_temporal("fecha")
    .configurar_split(test_size=0.2, validation_size=0.2, shuffle=False)
)
segmentos = splitter.ejecutar_segmentacion()
```

Si el DataFrame ya esta ordenado, puedes usar
`definir_orden_temporal(None, asumir_orden_actual=True)`. Sin una columna temporal
ni esa indicacion explicita, se genera un error para evitar cortes sobre filas
desordenadas. Los notebooks antiguos que omitian el orden deben actualizar esa
llamada. Con `modo="auto"`, la estratificacion se desactiva si `shuffle=False`
o si se separan grupos; una peticion explicita incompatible genera un error.

### Consultas

- `BahamutSplit`: clase principal en estilo PEP 8.
- `Bahamut_split`: alias compatible con el notebook original.
- `variables_disponibles_para_split()`: inventario de columnas candidatas y configuracion activa.
- `tabla_variables_split()`: version tabular del inventario anterior.
- `diagnostico_split()`: resumen operativo antes de ejecutar la segmentacion.
- `parametros_para_split_manual()`: parametros manuales reproducibles para reconstruir el split.

## Estado del proyecto

Este repositorio queda preparado para seguir anadiendo mas modulos de ML sobre una base de paquete Python con tests.
