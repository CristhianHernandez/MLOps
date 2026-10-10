"""Dos ejecuciones con los mismos argumentos deben producir exactamente el mismo resultado."""

import numpy as np
import pandas as pd

from diabetes_mlops_v2.config import ConfigModelo, ConfigParticion
from diabetes_mlops_v2.dataset import huella_datos
from diabetes_mlops_v2.features import dividir_datos, separar_x_y
from diabetes_mlops_v2.modeling.evaluate import (
    evaluar_referencia,
    predecir_probabilidades,
)
from diabetes_mlops_v2.modeling.train import fijar_semilla, seleccionar_c

CONFIG_MODELO = ConfigModelo(c_candidatos=(0.01, 1.0), solver="saga")  # saga es estocástico


def ejecutar(datos: pd.DataFrame, semilla: int):
    """Partición + selección de C + referencia, como en una corrida real."""
    particiones = dividir_datos(*separar_x_y(datos), ConfigParticion(), semilla)
    modelo, mejor_c, tabla = seleccionar_c(particiones, CONFIG_MODELO, semilla)
    prob_test = predecir_probabilidades(modelo, particiones.X_test)
    return particiones, mejor_c, tabla, prob_test, evaluar_referencia(particiones, semilla)


def test_misma_semilla_produce_el_mismo_resultado(datos_sinteticos):
    a = ejecutar(datos_sinteticos, semilla=42)
    b = ejecutar(datos_sinteticos, semilla=42)
    assert a[0].X_test.index.equals(b[0].X_test.index)
    assert a[1] == b[1]
    pd.testing.assert_frame_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])
    assert a[4] == b[4]


def test_otra_semilla_cambia_la_particion(datos_sinteticos):
    a = ejecutar(datos_sinteticos, semilla=42)
    b = ejecutar(datos_sinteticos, semilla=7)
    assert not a[0].X_test.index.equals(b[0].X_test.index)


def test_fijar_semilla_fija_los_generadores_globales():
    fijar_semilla(42)
    a = np.random.rand(3)
    fijar_semilla(42)
    np.testing.assert_array_equal(a, np.random.rand(3))


def test_huella_datos_identifica_el_contenido(tmp_path):
    ruta = tmp_path / "datos.csv"
    ruta.write_text("a,b\n1,2\n")
    huella = huella_datos(ruta)
    assert huella == huella_datos(ruta) and len(huella) == 64
    ruta.write_text("a,b\n1,3\n")
    assert huella_datos(ruta) != huella
