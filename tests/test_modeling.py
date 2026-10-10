import numpy as np
import pandas as pd
import pytest

from diabetes_mlops_v2.config import ConfigModelo, ConfigParticion
from diabetes_mlops_v2.features import dividir_datos, separar_x_y
from diabetes_mlops_v2.modeling.evaluate import (
    aplicar_umbral,
    calcular_metricas,
    evaluar_referencia,
    seleccionar_umbral,
)
from diabetes_mlops_v2.modeling.train import seleccionar_c


@pytest.fixture
def particiones(datos_sinteticos):
    return dividir_datos(*separar_x_y(datos_sinteticos), ConfigParticion(), semilla=42)


def test_aplicar_umbral():
    assert aplicar_umbral(np.array([0.2, 0.5, 0.8]), 0.5).tolist() == [0, 1, 1]


def test_seleccionar_umbral_garantiza_el_recall_minimo():
    y = pd.Series([0, 0, 0, 1, 1, 1, 1])
    prob = np.array([0.1, 0.3, 0.6, 0.2, 0.5, 0.7, 0.9])
    umbral = seleccionar_umbral(y, prob, recall_minimo=0.75)
    assert umbral == 0.5
    assert calcular_metricas(y, prob, umbral)["recall"] >= 0.75


def test_calcular_metricas_con_predicciones_perfectas():
    y = pd.Series([0, 0, 1, 1])
    metricas = calcular_metricas(y, np.array([0.1, 0.2, 0.8, 0.9]), umbral=0.5)
    assert all(valor == 1.0 for valor in metricas.values())


def test_seleccionar_c_elige_el_mejor_en_validacion_y_supera_al_azar(particiones):
    config = ConfigModelo(c_candidatos=(0.01, 1.0))
    modelo, mejor_c, tabla = seleccionar_c(particiones, config, semilla=42)
    assert mejor_c == tabla["roc_auc_val"].idxmax()
    assert list(tabla.index) == [0.01, 1.0]
    assert tabla.loc[mejor_c, "roc_auc_val"] > evaluar_referencia(particiones, 42)["roc_auc"]
    assert modelo.named_steps["modelo"].C == mejor_c
