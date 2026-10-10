"""Configuración central del proyecto: rutas, definición de variables y parámetros por defecto.

Es la única fuente de verdad para constantes y parámetros; el resto de los módulos
y el notebook de exploración los importan desde aquí.
"""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

# Rutas del proyecto (estructura Cookiecutter Data Science)
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Variables de entorno locales (por ejemplo, MLFLOW_TRACKING_URI); ver .env.example
load_dotenv(PROJ_ROOT / ".env")

# Fuente de datos
UCI_DATASET_ID = 891
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"
RAW_DATA_FILE = RAW_DATA_DIR / "cdc_diabetes_health_indicators.csv"

# MLflow: servidor de tracking y experimento en el que se agrupan las corridas
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENTO = os.environ.get("MLFLOW_EXPERIMENT_NAME", "diabetes-cdc-logreg")

# Variables agrupadas por su tipo real (pandas las carga todas como int64)
TARGET = "Diabetes_binary"
NUMERICAS = ["BMI", "MentHlth", "PhysHlth"]
ORDINALES = ["GenHlth", "Age", "Education", "Income"]
BINARIAS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]
# Orden de las columnas tal como las publica UCI (el modelo espera este orden)
PREDICTORAS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

# Rango válido de cada variable (BMI no tiene rango documentado: se usa un rango amplio de captura)
ESQUEMA = {col: (0, 1) for col in BINARIAS + [TARGET]}
ESQUEMA.update(
    {
        "BMI": (10, 100),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
        "GenHlth": (1, 5),
        "Age": (1, 13),
        "Education": (1, 6),
        "Income": (1, 8),
    }
)


@dataclass(frozen=True)
class ConfigLimpieza:
    """Decisiones de limpieza documentadas en el notebook de exploración."""

    eliminar_duplicados: bool = False
    bmi_max_plausible: int = 70
    tipo_datos: str = "int8"


@dataclass(frozen=True)
class ConfigParticion:
    """Partición estratificada en entrenamiento / validación / prueba."""

    test_size: float = 0.15
    val_size: float = 0.15


@dataclass(frozen=True)
class ConfigModelo:
    """Hiperparámetros de la regresión logística y valores de C a evaluar en validación."""

    c_candidatos: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    solver: str = "lbfgs"


@dataclass(frozen=True)
class ConfigEvaluacion:
    """Criterio para elegir el umbral de decisión en validación."""

    recall_minimo: float = 0.80


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """Configuración completa de una corrida de entrenamiento."""

    semilla: int = 42
    limpieza: ConfigLimpieza = field(default_factory=ConfigLimpieza)
    particion: ConfigParticion = field(default_factory=ConfigParticion)
    modelo: ConfigModelo = field(default_factory=ConfigModelo)
    evaluacion: ConfigEvaluacion = field(default_factory=ConfigEvaluacion)

    def como_diccionario(self) -> dict:
        """Configuración como diccionario plano (por ejemplo, {'modelo.max_iter': 1000})."""
        plano = {"semilla": self.semilla}
        for seccion, valores in asdict(self).items():
            if isinstance(valores, dict):
                plano.update({f"{seccion}.{k}": v for k, v in valores.items()})
        return plano
