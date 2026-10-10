"""Evaluación del modelo: probabilidades, umbral de decisión y métricas."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from diabetes_mlops_v2.features import Particiones


def predecir_probabilidades(modelo, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la clase positiva (diabetes)."""
    return modelo.predict_proba(X)[:, 1]


def aplicar_umbral(probabilidades: np.ndarray, umbral: float) -> np.ndarray:
    """Convierte probabilidades en predicciones 0/1 con el umbral indicado."""
    return (probabilidades >= umbral).astype(int)


def seleccionar_umbral(y: pd.Series, probabilidades: np.ndarray, recall_minimo: float) -> float:
    """Umbral más alto que mantiene un recall de al menos `recall_minimo`.

    Prioriza no dejar casos sin detectar (tamizaje) generando el menor número
    posible de falsas alarmas. Debe calcularse con el conjunto de validación.
    """
    _, recall, umbrales = precision_recall_curve(y, probabilidades)
    candidatos = umbrales[recall[:-1] >= recall_minimo]
    if candidatos.size == 0:
        raise ValueError(f"Ningún umbral alcanza un recall de {recall_minimo}")
    return float(candidatos.max())


def calcular_metricas(y: pd.Series, probabilidades: np.ndarray, umbral: float) -> dict[str, float]:
    """Métricas de ranking (independientes del umbral) y de decisión (con el umbral dado)."""
    predicciones = aplicar_umbral(probabilidades, umbral)
    tn, fp, _, _ = confusion_matrix(y, predicciones).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probabilidades)),
        "pr_auc": float(average_precision_score(y, probabilidades)),
        "recall": float(recall_score(y, predicciones)),
        "precision": float(precision_score(y, predicciones, zero_division=0)),
        "f1": float(f1_score(y, predicciones)),
        "especificidad": float(tn / (tn + fp)),
        "accuracy": float(accuracy_score(y, predicciones)),
    }


def evaluar_referencia(particiones: Particiones, semilla: int) -> dict[str, float]:
    """Métricas en prueba de un modelo que predice al azar respetando la proporción de clases.

    Sirve como piso de referencia: el modelo entrenado debe superarlo con claridad.
    """
    dummy = DummyClassifier(strategy="stratified", random_state=semilla)
    dummy.fit(particiones.X_train, particiones.y_train)
    prob_test = predecir_probabilidades(dummy, particiones.X_test)
    return calcular_metricas(particiones.y_test, prob_test, umbral=0.5)


def obtener_coeficientes(modelo: Pipeline) -> pd.Series:
    """Coeficientes de la regresión logística (sobre variables estandarizadas), ordenados."""
    coeficientes = modelo.named_steps["modelo"].coef_[0]
    return pd.Series(coeficientes, index=modelo.feature_names_in_).sort_values()
