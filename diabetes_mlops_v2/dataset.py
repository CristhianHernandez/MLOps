"""Carga, validación y limpieza de los datos.

Uso como script (descarga la copia local de los datos crudos):
    python -m diabetes_mlops_v2.dataset
    python -m diabetes_mlops_v2.dataset --forzar-descarga
"""

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

from diabetes_mlops_v2.config import (
    ESQUEMA,
    PREDICTORAS,
    RAW_DATA_FILE,
    TARGET,
    UCI_DATASET_ID,
    ConfigLimpieza,
)

logger = logging.getLogger(__name__)


def descargar_datos(uci_id: int = UCI_DATASET_ID) -> pd.DataFrame:
    """Descarga el dataset desde UCI y une predictoras y variable objetivo en un solo DataFrame."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=uci_id)
    return pd.concat([dataset.data.features, dataset.data.targets], axis=1)


def cargar_datos(ruta: Path = RAW_DATA_FILE, forzar_descarga: bool = False) -> pd.DataFrame:
    """Carga los datos crudos desde una copia local; si no existe, los descarga y la guarda.

    Trabajar con una copia local evita depender de la disponibilidad de UCI y garantiza
    que todas las corridas usen exactamente los mismos datos.
    """
    if forzar_descarga or not ruta.exists():
        logger.info("Descargando dataset %s desde UCI", UCI_DATASET_ID)
        datos = descargar_datos()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        datos.to_csv(ruta, index=False)
        logger.info("Copia local guardada en %s", ruta)
    datos = pd.read_csv(ruta)
    logger.info("Datos cargados: %s registros, %s columnas", f"{len(datos):,}", datos.shape[1])
    return datos


def huella_datos(ruta: Path = RAW_DATA_FILE) -> str:
    """SHA-256 del archivo de datos: identifica exactamente qué datos usó una corrida."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def validar_esquema(datos: pd.DataFrame, esquema: dict = ESQUEMA) -> None:
    """Verifica columnas, valores faltantes y rangos válidos; lanza ValueError si algo no cumple."""
    faltantes = set(PREDICTORAS + [TARGET]) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    if datos.isna().any().any():
        raise ValueError(f"Hay valores faltantes en: {datos.columns[datos.isna().any()].tolist()}")
    fuera_de_rango = {
        col: int((~datos[col].between(minimo, maximo)).sum())
        for col, (minimo, maximo) in esquema.items()
        if not datos[col].between(minimo, maximo).all()
    }
    if fuera_de_rango:
        raise ValueError(f"Valores fuera de dominio (columna: cantidad): {fuera_de_rango}")


def limpiar_datos(datos: pd.DataFrame, config: ConfigLimpieza = ConfigLimpieza()) -> pd.DataFrame:
    """Aplica las decisiones de limpieza documentadas en el notebook de exploración.

    1. Valida el esquema (sin faltantes ni valores fuera de dominio).
    2. Conserva los duplicados por defecto: son personas distintas con el mismo perfil.
    3. Elimina registros con BMI no plausible (errores de captura).
    4. Convierte las columnas a un tipo compacto.
    """
    validar_esquema(datos)
    n_inicial = len(datos)

    if config.eliminar_duplicados:
        datos = datos.drop_duplicates()
    datos = datos[datos["BMI"] <= config.bmi_max_plausible]
    datos = datos.astype(config.tipo_datos).reset_index(drop=True)

    eliminados = n_inicial - len(datos)
    logger.info(
        "Limpieza: %s registros eliminados (%.2f%%)",
        f"{eliminados:,}",
        eliminados / n_inicial * 100,
    )
    return datos


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Descarga desde UCI la copia local de los datos crudos y valida su esquema."
    )
    parser.add_argument(
        "--forzar-descarga",
        action="store_true",
        help=f"Vuelve a descargar aunque ya exista {RAW_DATA_FILE.name}.",
    )
    args = parser.parse_args(argv)
    validar_esquema(cargar_datos(forzar_descarga=args.forzar_descarga))
    logger.info("Esquema válido: %s", RAW_DATA_FILE)


if __name__ == "__main__":
    main()
