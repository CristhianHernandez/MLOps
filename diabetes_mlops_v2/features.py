"""Preparación de variables: separación X/y y partición entrenamiento / validación / prueba."""

from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_mlops_v2.config import PREDICTORAS, TARGET, ConfigParticion


class Particiones(NamedTuple):
    """Conjuntos de entrenamiento, validación y prueba."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def separar_x_y(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las predictoras (en el orden esperado por el modelo) de la variable objetivo."""
    return datos[PREDICTORAS], datos[TARGET]


def dividir_datos(
    X: pd.DataFrame, y: pd.Series, config: ConfigParticion, semilla: int
) -> Particiones:
    """Divide en entrenamiento / validación / prueba de forma estratificada.

    Primero separa el conjunto de prueba y después, del resto, el de validación,
    de modo que ambos representen el porcentaje indicado del total.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=semilla
    )
    val_relativo = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relativo, stratify=y_temp, random_state=semilla
    )
    return Particiones(X_train, X_val, X_test, y_train, y_val, y_test)


def resumir_particiones(particiones: Particiones) -> pd.DataFrame:
    """Tamaño y proporción de la clase positiva en cada partición."""
    conjuntos = {
        "entrenamiento": particiones.y_train,
        "validación": particiones.y_val,
        "prueba": particiones.y_test,
    }
    total = sum(len(y) for y in conjuntos.values())
    return pd.DataFrame(
        {
            "registros": {nombre: len(y) for nombre, y in conjuntos.items()},
            "pct_total": {nombre: len(y) / total * 100 for nombre, y in conjuntos.items()},
            "pct_positivos": {nombre: y.mean() * 100 for nombre, y in conjuntos.items()},
        }
    ).round(2)
