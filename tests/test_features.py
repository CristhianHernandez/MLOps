import pytest

from diabetes_mlops_v2.config import PREDICTORAS, ConfigParticion
from diabetes_mlops_v2.features import dividir_datos, separar_x_y


def test_separar_x_y_respeta_el_orden_de_predictoras(datos_sinteticos):
    X, y = separar_x_y(datos_sinteticos.iloc[:, ::-1])
    assert list(X.columns) == PREDICTORAS
    assert y.name == "Diabetes_binary"


def test_dividir_datos_respeta_proporciones_y_estratifica(datos_sinteticos):
    X, y = separar_x_y(datos_sinteticos)
    p = dividir_datos(X, y, ConfigParticion(test_size=0.2, val_size=0.1), semilla=42)
    n = len(X)
    assert len(p.X_test) == pytest.approx(0.2 * n, abs=1)
    assert len(p.X_val) == pytest.approx(0.1 * n, abs=1)
    assert len(p.X_train) + len(p.X_val) + len(p.X_test) == n
    for y_parte in (p.y_train, p.y_val, p.y_test):
        assert y_parte.mean() == pytest.approx(y.mean(), abs=0.01)


def test_dividir_datos_es_reproducible_con_la_misma_semilla(datos_sinteticos):
    X, y = separar_x_y(datos_sinteticos)
    a = dividir_datos(X, y, ConfigParticion(), semilla=7)
    b = dividir_datos(X, y, ConfigParticion(), semilla=7)
    assert a.X_test.index.equals(b.X_test.index)
