"""Plumbing compartido para las capas interactivas de notebook.

Copia vendorizada: manten los cuatro ficheros sincronizados.

    Fuente canonica : bahamut/src/bahamut/_notebook.py
    Copias          : eden/src/eden/_notebook.py
                      fenrir/_notebook.py

Los paquetes de la familia (bahamut, eden, fenrir, bahamut-tuning) se instalan
por separado, asi que no pueden compartir una dependencia interna sin publicar
un quinto paquete. Lo que si pueden compartir es *el mismo fichero*: antes cada
uno llevaba su propia version ligeramente distinta de estas cuatro funciones
(`_require_widgets`, `_publish_notebook_bindings`, `_round_frame`,
`_require_matplotlib`), con mensajes de error y comportamientos divergentes.

Regla de oro: este modulo no importa nada del paquete que lo aloja. Copiarlo de
un paquete a otro tiene que ser un `cp` sin editar.
"""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

try:  # pragma: no cover - depende del extra `notebook`
    import ipywidgets as widgets
    from IPython import get_ipython
    from IPython.display import clear_output, display
except Exception:  # pragma: no cover - camino sin el extra instalado
    widgets = None
    get_ipython = None
    clear_output = None
    display = None


__all__ = [
    "clear_output",
    "display",
    "get_ipython",
    "notebook_stack_available",
    "publish_notebook_bindings",
    "require_matplotlib_pyplot",
    "require_widgets",
    "round_frame",
    "widgets",
]


def notebook_stack_available() -> bool:
    """True cuando ipywidgets e IPython estan disponibles."""

    return widgets is not None and clear_output is not None and display is not None


def require_widgets(package: str, extra: str = "notebook") -> None:
    """Falla con un mensaje accionable si falta el extra de notebook."""

    if notebook_stack_available():
        return
    raise ImportError(
        f"La capa interactiva de {package} necesita ipywidgets e IPython. "
        f"Instala el extra correspondiente: pip install '{package}[{extra}]'."
    )


def require_matplotlib_pyplot():
    """Devuelve `matplotlib.pyplot` o falla con un mensaje accionable."""

    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "matplotlib es necesario para las funciones graficas. "
            "Instalalo con 'pip install matplotlib' o usa las salidas tabulares."
        ) from exc


def publish_notebook_bindings(**values: Any) -> None:
    """Publica variables en el namespace del notebook, si hay uno."""

    if get_ipython is None:
        return

    shell = get_ipython()
    if shell is None or not hasattr(shell, "user_ns"):
        return

    shell.user_ns.update(values)


def round_frame(frame: pd.DataFrame, round_digits: int | None) -> pd.DataFrame:
    """Redondea solo las columnas numericas, sin tocar el frame original."""

    if round_digits is None:
        return frame

    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include="number").columns
    if len(numeric_columns):
        rounded.loc[:, numeric_columns] = rounded.loc[:, numeric_columns].round(int(round_digits))
    return rounded
