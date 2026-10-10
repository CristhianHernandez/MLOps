import pytest

from diabetes_mlops_v2.config import ConfigLimpieza
from diabetes_mlops_v2.dataset import limpiar_datos, validar_esquema


def test_validar_esquema_acepta_datos_validos(datos_sinteticos):
    validar_esquema(datos_sinteticos)


def test_validar_esquema_detecta_columnas_faltantes(datos_sinteticos):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_esquema(datos_sinteticos.drop(columns=["BMI"]))


def test_validar_esquema_detecta_faltantes(datos_sinteticos):
    datos = datos_sinteticos.astype(float)
    datos.loc[0, "Age"] = None
    with pytest.raises(ValueError, match="valores faltantes"):
        validar_esquema(datos)


def test_validar_esquema_detecta_valores_fuera_de_dominio(datos_sinteticos):
    datos = datos_sinteticos.copy()
    datos.loc[0, "GenHlth"] = 9
    with pytest.raises(ValueError, match="fuera de dominio"):
        validar_esquema(datos)


def test_limpiar_datos_elimina_bmi_no_plausible_y_compacta_tipos(datos_sinteticos):
    datos = datos_sinteticos.copy()
    datos.loc[:9, "BMI"] = 95
    limpios = limpiar_datos(datos, ConfigLimpieza(bmi_max_plausible=70))
    assert len(limpios) == len(datos) - 10
    assert limpios["BMI"].max() <= 70
    assert (limpios.dtypes == "int8").all()


def test_limpiar_datos_conserva_duplicados_por_defecto(datos_sinteticos):
    datos = datos_sinteticos.iloc[[0, 0, 1]]
    assert len(limpiar_datos(datos)) == 3
    assert len(limpiar_datos(datos, ConfigLimpieza(eliminar_duplicados=True))) == 2
