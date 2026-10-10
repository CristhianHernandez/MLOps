"""Datos sintéticos con el mismo esquema que el dataset real (no requieren descargar nada)."""

import numpy as np
import pandas as pd
import pytest

from diabetes_mlops_v2.config import ESQUEMA, PREDICTORAS, TARGET


@pytest.fixture
def datos_sinteticos() -> pd.DataFrame:
    """2,000 registros válidos con ~15 % de positivos; el riesgo crece con BMI y GenHlth."""
    rng = np.random.default_rng(0)
    n = 2_000
    datos = pd.DataFrame(
        {col: rng.integers(minimo, maximo + 1, n) for col, (minimo, maximo) in ESQUEMA.items()}
    )
    datos["BMI"] = rng.integers(15, 60, n)
    riesgo = 0.08 * (datos["BMI"] - 35) + 0.8 * (datos["GenHlth"] - 3) - 1.9
    datos[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-riesgo))).astype(int)
    return datos[PREDICTORAS + [TARGET]]
