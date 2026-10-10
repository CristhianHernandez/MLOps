import pytest

from diabetes_mlops_v2.config import ConfigEntrenamiento
from diabetes_mlops_v2.modeling.train import config_desde_argumentos, construir_parser
from diabetes_mlops_v2.tracking import nombre_corrida


def configurar(*argv):
    return config_desde_argumentos(construir_parser().parse_args(list(argv)))


def test_sin_argumentos_usa_la_configuracion_por_defecto():
    assert configurar() == ConfigEntrenamiento()


def test_argumentos_personalizados():
    config = configurar("--C", "0.1", "--class-weight", "none", "--max-iter", "500", "--seed", "7")
    assert config.modelo.c_candidatos == (0.1,)
    assert config.modelo.class_weight is None
    assert config.modelo.max_iter == 500
    assert config.semilla == 7


@pytest.mark.parametrize(
    "argv", [["--C", "-1"], ["--max-iter", "0"], ["--test-size", "1.5"], ["--solver", "adam"]]
)
def test_rechaza_valores_invalidos(argv):
    with pytest.raises(SystemExit):
        construir_parser().parse_args(argv)


def test_rechaza_particiones_que_no_dejan_datos_de_entrenamiento():
    with pytest.raises(ValueError, match="menor que 1"):
        configurar("--test-size", "0.6", "--val-size", "0.5")


def test_nombre_corrida_describe_los_hiperparametros():
    assert nombre_corrida(ConfigEntrenamiento()) == "logreg-grid-balanced-lbfgs-seed42"
    assert nombre_corrida(configurar("--C", "0.1", "--class-weight", "none", "--seed", "7")) == (
        "logreg-C0.1-none-lbfgs-seed7"
    )
