"""Entrenamiento del modelo base de detección de diabetes (CDC Diabetes Health Indicators, UCI 891).

Flujo: datos → limpieza → partición train/val/test → entrenamiento y selección de C →
umbral de decisión → evaluación en prueba → guardado de modelo, reportes y gráficas.
Cada corrida se registra en MLflow Tracking (parámetros, métricas, artefactos y modelo).

Uso (requiere el servidor de MLflow activo; ver `make mlflow-server`):
    python -m diabetes_mlops_v2.modeling.train                    # valores por defecto
    python -m diabetes_mlops_v2.modeling.train --C 0.1 --seed 7   # hiperparámetros personalizados
    python -m diabetes_mlops_v2.modeling.train --help             # todos los argumentos
"""

import argparse
import json
import logging
from pathlib import Path
import random
from typing import NamedTuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diabetes_mlops_v2 import plots, tracking
from diabetes_mlops_v2.config import (
    MLFLOW_EXPERIMENTO,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REPORTS_DIR,
    ConfigEntrenamiento,
    ConfigEvaluacion,
    ConfigLimpieza,
    ConfigModelo,
    ConfigParticion,
)
from diabetes_mlops_v2.dataset import cargar_datos, huella_datos, limpiar_datos
from diabetes_mlops_v2.features import (
    Particiones,
    dividir_datos,
    resumir_particiones,
    separar_x_y,
)
from diabetes_mlops_v2.modeling.evaluate import (
    aplicar_umbral,
    calcular_metricas,
    evaluar_referencia,
    obtener_coeficientes,
    predecir_probabilidades,
    seleccionar_umbral,
)

logger = logging.getLogger(__name__)

SOLVERS = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]


# =============================================================================
# Modelo: construcción y selección del hiperparámetro C
# =============================================================================


def crear_modelo(C: float, config: ConfigModelo, semilla: int) -> Pipeline:
    """Pipeline de estandarización + regresión logística.

    El escalador va dentro del Pipeline para que se ajuste solo con entrenamiento
    (sin fuga de información).
    """
    regresion = LogisticRegression(
        C=C,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        solver=config.solver,
        random_state=semilla,
    )
    return Pipeline([("escalador", StandardScaler()), ("modelo", regresion)])


def seleccionar_c(
    particiones: Particiones, config: ConfigModelo, semilla: int
) -> tuple[Pipeline, float, pd.DataFrame]:
    """Entrena un modelo por cada C candidato y elige el de mayor ROC-AUC en validación.

    Devuelve el modelo ya entrenado con el mejor C (no se reentrena), el valor de C
    y la tabla de resultados de todos los candidatos.
    """
    resultados, modelos = [], {}
    for C in config.c_candidatos:
        modelo = crear_modelo(C, config, semilla).fit(particiones.X_train, particiones.y_train)
        modelos[C] = modelo
        prob_train = predecir_probabilidades(modelo, particiones.X_train)
        prob_val = predecir_probabilidades(modelo, particiones.X_val)
        resultados.append(
            {
                "C": C,
                "roc_auc_train": roc_auc_score(particiones.y_train, prob_train),
                "roc_auc_val": roc_auc_score(particiones.y_val, prob_val),
            }
        )
    tabla = pd.DataFrame(resultados).set_index("C")
    mejor_c = float(tabla["roc_auc_val"].idxmax())
    logger.info("Selección de C (ROC-AUC):\n%s", tabla.round(4).to_string())
    return modelos[mejor_c], mejor_c, tabla


# =============================================================================
# Persistencia local: modelo, reportes y gráficas de la última corrida
# =============================================================================


def guardar_resultados(
    modelo: Pipeline,
    resumen: dict,
    tabla_c: pd.DataFrame,
    dir_modelos: Path = MODELS_DIR,
    dir_reportes: Path = REPORTS_DIR,
) -> list[Path]:
    """Guarda el modelo, el resumen de la corrida (JSON) y la tabla de selección de C.

    Devuelve las rutas de los reportes para registrarlos como artefactos.
    """
    dir_modelos.mkdir(parents=True, exist_ok=True)
    dir_reportes.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, dir_modelos / "modelo_base.joblib")
    ruta_resumen = dir_reportes / "resultados.json"
    ruta_tabla_c = dir_reportes / "seleccion_c.csv"
    ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    tabla_c.to_csv(ruta_tabla_c)
    logger.info("Modelo y reportes guardados en %s y %s", dir_modelos, dir_reportes)
    return [ruta_resumen, ruta_tabla_c]


def generar_graficas(
    modelo: Pipeline,
    particiones: Particiones,
    prob_test: np.ndarray,
    metricas: dict,
    umbral: float,
) -> list[Path]:
    """Genera y guarda las gráficas de evaluación sobre el conjunto de prueba."""
    plots.configurar_estilo()
    pred_test = aplicar_umbral(prob_test, umbral)
    figuras = {
        "matriz_confusion": plots.graficar_matriz_confusion(particiones.y_test, pred_test, umbral),
        "curvas_roc_pr": plots.graficar_curvas_roc_pr(
            particiones.y_test, prob_test, metricas, umbral
        ),
        "coeficientes": plots.graficar_coeficientes(obtener_coeficientes(modelo)),
    }
    return [plots.guardar_figura(fig, nombre) for nombre, fig in figuras.items()]


# =============================================================================
# Orquestación del flujo completo
# =============================================================================


class ResultadoEntrenamiento(NamedTuple):
    """Lo que produce una corrida de entrenamiento y se registra en MLflow."""

    modelo: Pipeline
    particiones: Particiones
    resumen: dict
    rutas_graficas: list[Path]
    rutas_reportes: list[Path]


def fijar_semilla(semilla: int) -> None:
    """Fija los generadores aleatorios globales de Python y NumPy.

    Cada componente aleatorio del flujo (partición, regresión logística y modelo de referencia)
    ya recibe `random_state=semilla` de forma explícita; fijar los generadores globales cubre
    además cualquier operación que no lo reciba.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def entrenar(config: ConfigEntrenamiento = ConfigEntrenamiento()) -> ResultadoEntrenamiento:
    """Flujo completo: datos → limpieza → partición → entrenamiento → umbral → evaluación."""
    fijar_semilla(config.semilla)

    # Datos
    datos = limpiar_datos(cargar_datos(), config.limpieza)
    particiones = dividir_datos(*separar_x_y(datos), config.particion, config.semilla)
    logger.info("Particiones:\n%s", resumir_particiones(particiones).to_string())

    # Entrenamiento y selección de C con validación
    modelo, mejor_c, tabla_c = seleccionar_c(particiones, config.modelo, config.semilla)

    # Umbral de decisión elegido con validación
    prob_val = predecir_probabilidades(modelo, particiones.X_val)
    umbral = seleccionar_umbral(particiones.y_val, prob_val, config.evaluacion.recall_minimo)

    # Evaluación final: una sola vez sobre prueba
    prob_test = predecir_probabilidades(modelo, particiones.X_test)
    metricas_test = calcular_metricas(particiones.y_test, prob_test, umbral)
    metricas_val = calcular_metricas(particiones.y_val, prob_val, umbral)
    metricas_dummy = evaluar_referencia(particiones, config.semilla)
    metricas = {
        **{f"test_{k}": v for k, v in metricas_test.items()},
        **{f"val_{k}": v for k, v in metricas_val.items()},
        "train_roc_auc": float(tabla_c.loc[mejor_c, "roc_auc_train"]),
        **{f"dummy_test_{k}": v for k, v in metricas_dummy.items()},
    }

    resumen = {
        "config": config.como_diccionario(),
        "mejor_c": mejor_c,
        "umbral": umbral,
        "datos_sha256": huella_datos(),
        "n_registros": {
            "train": len(particiones.y_train),
            "val": len(particiones.y_val),
            "test": len(particiones.y_test),
        },
        "metricas": metricas,
    }
    rutas_reportes = guardar_resultados(modelo, resumen, tabla_c)
    rutas_graficas = generar_graficas(modelo, particiones, prob_test, metricas_test, umbral)
    logger.info("Gráficas guardadas: %s", [r.name for r in rutas_graficas])
    logger.info(
        "C = %s | umbral = %.4f | ROC-AUC prueba = %.4f | recall prueba = %.4f",
        mejor_c,
        umbral,
        metricas["test_roc_auc"],
        metricas["test_recall"],
    )
    return ResultadoEntrenamiento(modelo, particiones, resumen, rutas_graficas, rutas_reportes)


# =============================================================================
# Argumentos de línea de comandos
# =============================================================================


def fraccion(valor: str) -> float:
    """Valida que el argumento sea un número entre 0 y 1 (exclusivo)."""
    numero = float(valor)
    if not 0 < numero < 1:
        raise argparse.ArgumentTypeError(f"debe estar entre 0 y 1 (exclusivo), se recibió {valor}")
    return numero


def positivo(tipo):
    """Crea un validador que convierte al tipo indicado y exige un valor mayor que cero."""

    def validar(valor: str):
        numero = tipo(valor)
        if numero <= 0:
            raise argparse.ArgumentTypeError(f"debe ser mayor que 0, se recibió {valor}")
        return numero

    return validar


def construir_parser() -> argparse.ArgumentParser:
    """Define los argumentos de línea de comandos.

    Los valores por defecto salen de las dataclasses de configuración (una sola fuente de verdad).
    """
    defaults = ConfigEntrenamiento()
    c_por_defecto = " ".join(map(str, defaults.modelo.c_candidatos))
    parser = argparse.ArgumentParser(
        prog="python -m diabetes_mlops_v2.modeling.train",
        description="Entrena y evalúa el modelo base (regresión logística) para detectar "
        "prediabetes/diabetes con el dataset CDC Diabetes Health Indicators, "
        "y registra la corrida en MLflow.",
        epilog="Ejemplos:\n"
        "  python -m diabetes_mlops_v2.modeling.train\n"
        "  python -m diabetes_mlops_v2.modeling.train --C 0.1 --max-iter 500 --seed 7\n"
        "  python -m diabetes_mlops_v2.modeling.train --C 0.01 0.1 1 --class-weight none "
        "--solver liblinear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    modelo = parser.add_argument_group("hiperparámetros del modelo (LogisticRegression)")
    modelo.add_argument(
        "--C",
        dest="c_candidatos",
        type=positivo(float),
        nargs="+",
        metavar="C",
        default=list(defaults.modelo.c_candidatos),
        help="Inverso de la fuerza de regularización (valores pequeños = modelo más simple). "
        "Con un solo valor se entrena ese modelo; con varios se elige el de mayor ROC-AUC "
        f"en validación. Por defecto: {c_por_defecto}",
    )
    modelo.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default=defaults.modelo.class_weight or "none",
        help="'balanced' da más peso a la clase minoritaria (diabetes) para compensar el "
        "desbalance ~86/14; 'none' trata ambas clases igual. "
        f"Por defecto: {defaults.modelo.class_weight}",
    )
    modelo.add_argument(
        "--max-iter",
        type=positivo(int),
        default=defaults.modelo.max_iter,
        help=f"Número máximo de iteraciones del optimizador. Por defecto: {defaults.modelo.max_iter}",
    )
    modelo.add_argument(
        "--solver",
        choices=SOLVERS,
        default=defaults.modelo.solver,
        help="Algoritmo de optimización de la regresión logística. "
        f"Por defecto: {defaults.modelo.solver}",
    )

    general = parser.add_argument_group("reproducibilidad")
    general.add_argument(
        "--seed",
        type=int,
        default=defaults.semilla,
        help="Semilla aleatoria para la partición de datos, el modelo y el modelo de referencia. "
        f"Por defecto: {defaults.semilla}",
    )

    flujo = parser.add_argument_group("datos y evaluación")
    flujo.add_argument(
        "--recall-minimo",
        type=fraccion,
        default=defaults.evaluacion.recall_minimo,
        help="Recall mínimo que debe alcanzar el umbral de decisión en validación (tamizaje). "
        f"Por defecto: {defaults.evaluacion.recall_minimo}",
    )
    flujo.add_argument(
        "--test-size",
        type=fraccion,
        default=defaults.particion.test_size,
        help=f"Proporción del total destinada a prueba. Por defecto: {defaults.particion.test_size}",
    )
    flujo.add_argument(
        "--val-size",
        type=fraccion,
        default=defaults.particion.val_size,
        help="Proporción del total destinada a validación. "
        f"Por defecto: {defaults.particion.val_size}",
    )
    flujo.add_argument(
        "--bmi-max",
        type=positivo(int),
        default=defaults.limpieza.bmi_max_plausible,
        help="BMI máximo plausible; los registros por encima se eliminan como errores de "
        f"captura. Por defecto: {defaults.limpieza.bmi_max_plausible}",
    )
    flujo.add_argument(
        "--eliminar-duplicados",
        action="store_true",
        help="Elimina filas duplicadas (por defecto se conservan: son personas distintas con "
        "el mismo perfil).",
    )

    seguimiento = parser.add_argument_group("seguimiento en MLflow")
    seguimiento.add_argument(
        "--tracking-uri",
        default=MLFLOW_TRACKING_URI,
        help="URI del servidor de MLflow Tracking (o variable MLFLOW_TRACKING_URI). "
        f"Por defecto: {MLFLOW_TRACKING_URI}",
    )
    seguimiento.add_argument(
        "--experimento",
        default=MLFLOW_EXPERIMENTO,
        help="Experimento de MLflow en el que se agrupan las corridas "
        f"(o variable MLFLOW_EXPERIMENT_NAME). Por defecto: {MLFLOW_EXPERIMENTO}",
    )
    seguimiento.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida. Por defecto se genera a partir de los hiperparámetros "
        "(por ejemplo, logreg-grid-balanced-lbfgs-seed42).",
    )
    return parser


def config_desde_argumentos(args: argparse.Namespace) -> ConfigEntrenamiento:
    """Convierte los argumentos de línea de comandos en la configuración de la corrida."""
    if args.test_size + args.val_size >= 1:
        raise ValueError("--test-size + --val-size debe ser menor que 1")
    return ConfigEntrenamiento(
        semilla=args.seed,
        limpieza=ConfigLimpieza(
            eliminar_duplicados=args.eliminar_duplicados, bmi_max_plausible=args.bmi_max
        ),
        particion=ConfigParticion(test_size=args.test_size, val_size=args.val_size),
        modelo=ConfigModelo(
            c_candidatos=tuple(args.c_candidatos),
            class_weight=None if args.class_weight == "none" else args.class_weight,
            max_iter=args.max_iter,
            solver=args.solver,
        ),
        evaluacion=ConfigEvaluacion(recall_minimo=args.recall_minimo),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = construir_parser()
    args = parser.parse_args(argv)
    try:
        config = config_desde_argumentos(args)
    except ValueError as error:
        parser.error(str(error))
    logger.info("Configuración de la corrida: %s", config.como_diccionario())

    tracking.configurar_mlflow(args.tracking_uri, args.experimento)
    run_name = args.run_name or tracking.nombre_corrida(config)
    with mlflow.start_run(run_name=run_name) as corrida:
        tracking.registrar_configuracion(config)
        tracking.registrar_resultados(entrenar(config))
    logger.info(
        "Corrida registrada en MLflow: %s (run_id = %s)",
        corrida.info.run_name,
        corrida.info.run_id,
    )


if __name__ == "__main__":
    main()
