"""Registro de las corridas de entrenamiento en MLflow Tracking."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import mlflow
from mlflow.models import infer_signature
import pandas as pd

from diabetes_mlops_v2.config import (
    TARGET,
    UCI_DATASET_ID,
    UCI_DATASET_URL,
    ConfigEntrenamiento,
)
from diabetes_mlops_v2.features import Particiones
from diabetes_mlops_v2.modeling.evaluate import predecir_probabilidades

if TYPE_CHECKING:
    from diabetes_mlops_v2.modeling.train import ResultadoEntrenamiento

logger = logging.getLogger(__name__)


def configurar_mlflow(tracking_uri: str, experimento: str) -> None:
    """Conecta con el servidor de tracking y agrupa las corridas en el experimento indicado."""
    mlflow.autolog(disable=True)  # se registra todo de forma explícita
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experimento)
    # Las columnas son enteras sin faltantes (validado en limpiar_datos): el aviso no aplica
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")
    logger.info("MLflow: %s | experimento '%s'", tracking_uri, experimento)


def nombre_corrida(config: ConfigEntrenamiento) -> str:
    """Nombre descriptivo de la corrida a partir de sus hiperparámetros.

    Por ejemplo, 'logreg-C0.1-balanced-lbfgs-seed42' o 'logreg-grid-...' si se evalúan varios C.
    """
    modelo = config.modelo
    c = "grid" if len(modelo.c_candidatos) > 1 else f"C{modelo.c_candidatos[0]:g}"
    return f"logreg-{c}-{modelo.class_weight or 'none'}-{modelo.solver}-seed{config.semilla}"


def registrar_configuracion(config: ConfigEntrenamiento) -> None:
    """Etiquetas y parámetros de la corrida.

    Se registran antes de entrenar para que una corrida fallida también los conserve.
    """
    mlflow.set_tags(
        {
            "stage": "experimentacion",
            "model_family": "LogisticRegression",
            "dataset": f"CDC Diabetes Health Indicators (UCI {UCI_DATASET_ID})",
            "entrypoint": "diabetes_mlops_v2.modeling.train",
        }
    )
    for nombre, valor in config.como_diccionario().items():
        mlflow.log_param(nombre, valor)


def registrar_datasets(particiones: Particiones) -> None:
    """Registra los conjuntos de entrenamiento, validación y prueba como entradas de la corrida."""
    conjuntos = {
        "train": (particiones.X_train, particiones.y_train),
        "validation": (particiones.X_val, particiones.y_val),
        "test": (particiones.X_test, particiones.y_test),
    }
    for contexto, (X, y) in conjuntos.items():
        dataset = mlflow.data.from_pandas(
            pd.concat([X, y], axis=1),
            targets=TARGET,
            name=f"diabetes-{contexto}",
            source=UCI_DATASET_URL,
        )
        mlflow.log_input(dataset, context=contexto)


def registrar_artefactos(rutas: list[Path], carpeta: str) -> None:
    """Sube los archivos indicados a una carpeta de artefactos de la corrida."""
    for ruta in rutas:
        mlflow.log_artifact(str(ruta), artifact_path=carpeta)


def registrar_modelo(modelo, X_ejemplo: pd.DataFrame, umbral: float) -> None:
    """Registra el pipeline entrenado con su firma (esquema de entrada/salida) y un ejemplo."""
    info_modelo = mlflow.sklearn.log_model(
        sk_model=modelo,
        name="modelo",
        signature=infer_signature(X_ejemplo, predecir_probabilidades(modelo, X_ejemplo)),
        input_example=X_ejemplo.head(5),
    )
    mlflow.set_logged_model_tags(
        info_modelo.model_id,
        {
            "umbral_decision": f"{umbral:.4f}",
            "descripcion": "StandardScaler + LogisticRegression; salida = probabilidad de diabetes",
        },
    )


def registrar_resultados(resultado: "ResultadoEntrenamiento") -> None:
    """Registra datasets, parámetros elegidos en validación, métricas, artefactos y modelo."""
    resumen = resultado.resumen
    registrar_datasets(resultado.particiones)

    mlflow.log_param("seleccion.mejor_c", resumen["mejor_c"])
    mlflow.log_param("seleccion.umbral_decision", round(resumen["umbral"], 4))
    mlflow.log_param("datos.sha256", resumen["datos_sha256"])
    for conjunto, n in resumen["n_registros"].items():
        mlflow.log_param(f"datos.n_{conjunto}", n)

    for nombre, valor in resumen["metricas"].items():
        mlflow.log_metric(nombre, valor)

    registrar_artefactos(resultado.rutas_graficas, "graficas")
    registrar_artefactos(resultado.rutas_reportes, "reportes")
    registrar_modelo(resultado.modelo, resultado.particiones.X_train, resumen["umbral"])
