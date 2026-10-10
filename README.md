# diabetes-mlops-v2

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelo base para detectar prediabetes/diabetes con el dataset
[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
(UCI 891, 253,680 registros, 21 variables). El modelo es un `Pipeline` de `StandardScaler` +
regresión logística; el hiperparámetro `C` y el umbral de decisión se eligen en validación
(el umbral garantiza un recall mínimo, porque el objetivo es tamizaje) y la evaluación final se
hace una sola vez sobre prueba. Cada corrida queda registrada en MLflow Tracking con sus
parámetros, métricas, artefactos y modelo.

## Requisitos

- Python **3.13** (el proyecto se validó con 3.13.15). En Windows se instala desde
  [python.org](https://www.python.org/downloads/) o con `winget install Python.Python.3.13`.
- Git y acceso a internet (para instalar las dependencias y, la primera vez, descargar los datos
  desde UCI).

Todas las dependencias fijadas tienen paquetes precompilados para macOS, Linux y Windows x64 con
Python 3.13: no se necesita compilador.

## Instalación y ejecución en macOS / Linux

```bash
git clone https://github.com/CristhianHernandez/MLOps.git
cd MLOps
python3.13 -m venv .venv
source .venv/bin/activate
make requirements          # instala las versiones exactas de requirements-dev.txt + el paquete
cp .env.example .env       # configuración local de MLflow (ver abajo)
```

`make requirements` equivale a:

```bash
python -m pip install -r requirements-dev.txt     # dependencias fijadas (incluye requirements.txt)
python -m pip install -e . --no-deps              # el paquete diabetes_mlops_v2 en modo editable
```

## Archivos de dependencias

| Archivo | Contenido |
|---|---|
| `requirements.txt` | Versiones **exactas** (`==`) de las 9 librerías que usa el proyecto (`mlflow==3.16.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3`, …) |
| `requirements-dev.txt` | `requirements.txt` + herramientas de desarrollo fijadas (`pytest`, `black`, `isort`, `flake8`, `mkdocs`) |
| `pyproject.toml` | Metadatos del paquete y rangos compatibles de las dependencias directas |

### Ejecución

**1. Levantar el servidor de MLflow** (en una terminal aparte, dejarla abierta):

```bash
make mlflow-server         # http://127.0.0.1:5000 (SQLite en mlflow.db, artefactos en mlartifacts/)
```

**2. Descargar y validar los datos** (solo la primera vez; quedan en `data/raw/`):

```bash
make data                  # = python -m diabetes_mlops_v2.dataset
```

**3. Entrenar** (cada ejecución crea una corrida en el experimento `diabetes-cdc-logreg`):

```bash
python -m diabetes_mlops_v2.modeling.train                       # valores por defecto
python -m diabetes_mlops_v2.modeling.train --C 0.1 --seed 7      # hiperparámetros personalizados
python -m diabetes_mlops_v2.modeling.train --help                # todos los argumentos
make train ARGS="--C 1 --solver liblinear"                       # equivalente con make
```

**4. Consultar los resultados** en http://127.0.0.1:5000 → experimento `diabetes-cdc-logreg`.
La última corrida también se guarda localmente en `models/modelo_base.joblib`,
`reports/resultados.json`, `reports/seleccion_c.csv` y `reports/figures/`.

## Instalación y ejecución en Windows

Windows no incluye `make`, así que se usan directamente los comandos que ejecuta cada regla del
`Makefile`. Los siguientes comandos son para **PowerShell**, desde la carpeta donde se quiera
clonar el repositorio.

**Instalación** (una sola vez):

```powershell
git clone https://github.com/CristhianHernandez/MLOps.git
cd MLOps
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

> Si `Activate.ps1` falla con *"la ejecución de scripts está deshabilitada en este sistema"*,
> permite los scripts solo para la terminal actual y vuelve a activar el entorno:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, seguido de
> `.venv\Scripts\Activate.ps1`. En `cmd.exe` el entorno se activa con `.venv\Scripts\activate.bat`.

**Terminal 1: servidor de MLflow** (activar el entorno y dejarla abierta):

```powershell
cd MLOps
.venv\Scripts\Activate.ps1
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

**Terminal 2: datos, entrenamiento y pruebas:**

```powershell
cd MLOps
.venv\Scripts\Activate.ps1
python -m diabetes_mlops_v2.dataset                                # descarga y valida los datos
python -m diabetes_mlops_v2.modeling.train                         # valores por defecto
python -m diabetes_mlops_v2.modeling.train --C 0.1 --seed 7        # hiperparámetros personalizados
python -m diabetes_mlops_v2.modeling.train --help                  # todos los argumentos
python -m pytest                                                   # pruebas
```

Los resultados se consultan igual, en http://127.0.0.1:5000.

### Equivalencia entre `make` y los comandos directos

| Regla de `make` | Comando equivalente (cualquier sistema, con el entorno activado) |
|---|---|
| `make requirements` | `python -m pip install -r requirements-dev.txt` y `python -m pip install -e . --no-deps` |
| `make mlflow-server` | `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000` |
| `make data` | `python -m diabetes_mlops_v2.dataset` |
| `make train ARGS="..."` | `python -m diabetes_mlops_v2.modeling.train ...` |
| `make test` | `python -m pytest` |
| `make lint` | `flake8 diabetes_mlops_v2` · `isort --check --diff diabetes_mlops_v2` · `black --check diabetes_mlops_v2` |

## Argumentos del script de entrenamiento

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--C C [C ...]` | `0.001 0.01 0.1 1.0 10.0` | Inverso de la regularización. Con varios valores se elige el de mayor ROC-AUC en validación |
| `--class-weight {balanced,none}` | `balanced` | Peso de las clases (compensa el desbalance ~86/14) |
| `--max-iter` | `1000` | Iteraciones máximas del optimizador |
| `--solver` | `lbfgs` | `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag` o `saga` |
| `--seed` | `42` | Semilla aleatoria (ver [Reproducibilidad](#reproducibilidad)) |
| `--recall-minimo` | `0.8` | Recall mínimo que debe cumplir el umbral de decisión en validación |
| `--test-size` / `--val-size` | `0.15` / `0.15` | Proporción del total para prueba y validación |
| `--bmi-max` | `70` | BMI máximo plausible (los registros por encima se eliminan) |
| `--eliminar-duplicados` | desactivado | Elimina filas duplicadas (por defecto se conservan) |
| `--tracking-uri` | `MLFLOW_TRACKING_URI` del `.env` | Servidor de MLflow |
| `--experimento` | `MLFLOW_EXPERIMENT_NAME` del `.env` | Experimento donde se agrupan las corridas |
| `--run-name` | generado, p. ej. `logreg-grid-balanced-lbfgs-seed42` | Nombre de la corrida |

## Configuración de MLflow (`.env`)

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=diabetes-cdc-logreg
```

> **macOS:** AirPlay Receiver también escucha en el puerto 5000. Si el entrenamiento falla con
> `error code 403`, el servidor de MLflow no está corriendo y quien responde es AirPlay: levanta
> el servidor (`make mlflow-server`) o usa otro puerto (`make mlflow-server MLFLOW_PORT=5001` y
> `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` en `.env`).

## Qué se registra en cada corrida

- **Parámetros** (`log_param`): semilla, limpieza, partición, hiperparámetros del modelo, `C` y
  umbral elegidos en validación, tamaño de cada partición y huella SHA-256 de los datos.
- **Métricas** (`log_metric`): ROC-AUC, PR-AUC, recall, precisión, F1, especificidad y accuracy en
  prueba y validación; ROC-AUC en entrenamiento; y las mismas métricas de un modelo al azar
  (`dummy_test_*`) como piso de referencia.
- **Artefactos** (`log_artifact`): `graficas/` (matriz de confusión, curvas ROC/PR, coeficientes)
  y `reportes/` (`resultados.json`, `seleccion_c.csv`).
- **Modelo** (`mlflow.sklearn.log_model`): el pipeline completo con firma y ejemplo de entrada.
- **Datasets** (`log_input`): los conjuntos de entrenamiento, validación y prueba.

## Reproducibilidad

Dos ejecuciones con los mismos argumentos producen **exactamente** el mismo resultado: mismas
particiones, mismo `C`, mismo umbral, mismas métricas y el mismo archivo de modelo (byte a byte).

**Semilla.** El argumento `--seed` (por defecto `42`, definido en `ConfigEntrenamiento` de
`config.py`) controla toda la aleatoriedad del flujo:

| Dónde | Cómo se usa la semilla |
|---|---|
| `features.dividir_datos` | `random_state` de las dos particiones estratificadas (`train_test_split`) |
| `modeling.train.crear_modelo` | `random_state` de `LogisticRegression` (relevante con `liblinear`, `sag` y `saga`) |
| `modeling.evaluate.evaluar_referencia` | `random_state` del `DummyClassifier` de referencia |
| `modeling.train.fijar_semilla` | Generadores globales de `random` y `numpy`, para cualquier operación que no reciba `random_state` |

La semilla se registra en MLflow como el parámetro `semilla` y forma parte del nombre de la
corrida (`...-seed42`), de modo que cualquier resultado se puede volver a generar.

**Datos.** Todas las corridas leen la misma copia local `data/raw/cdc_diabetes_health_indicators.csv`
(no se vuelve a descargar de UCI en cada ejecución). Su huella SHA-256 se registra como el
parámetro `datos.sha256`: si dos corridas tienen la misma huella y los mismos parámetros, usaron
exactamente los mismos datos.

**Entorno.** `requirements.txt` fija la versión exacta de cada librería y `pyproject.toml` exige
Python 3.13. Además, el modelo registrado en MLflow incluye su propio `requirements.txt` y
`python_env.yaml`.

**Cómo comprobarlo:**

```bash
make test        # incluye tests/test_reproducibilidad.py (mismas particiones, modelo y métricas)
```

O manualmente: ejecuta dos veces el mismo comando, por ejemplo
`python -m diabetes_mlops_v2.modeling.train --seed 123 --run-name repro-1` y `--run-name repro-2`,
selecciona ambas corridas en MLflow → **Compare** y activa *Show diff only*: ningún parámetro
(salvo el nombre) ni métrica difiere.

## Desarrollo

```bash
make test        # pruebas unitarias (datos sintéticos; no requieren MLflow ni descargar datos)
make lint        # flake8, isort y black en modo verificación
make format      # aplica isort y black
make help        # lista de todas las reglas
```

## Organización del proyecto

```
├── Makefile             <- Comandos: requirements, data, train, mlflow-server, test, lint
├── README.md            <- Este documento
├── requirements.txt     <- Versiones exactas de las dependencias de ejecución
├── requirements-dev.txt <- requirements.txt + herramientas de desarrollo fijadas
├── pyproject.toml       <- Metadatos del paquete y configuración de black/isort/pytest
├── setup.cfg            <- Configuración de flake8
├── .env.example         <- Plantilla de variables de entorno (copiar como .env)
├── data
│   └── raw              <- Datos originales e inmutables (descargados con `make data`)
├── docs                 <- Proyecto mkdocs
├── models               <- Último modelo entrenado (cada corrida se versiona en MLflow)
├── notebooks
│   └── 1.0-ch-exploracion-datos.ipynb  <- Solo exploración (EDA y decisiones de limpieza)
├── references           <- Material de referencia (notebook baseline de la entrega anterior)
├── reports              <- Reportes y figuras de la última corrida
├── tests                <- Pruebas unitarias y de reproducibilidad
└── diabetes_mlops_v2    <- Código fuente
    ├── config.py        <- Rutas, variables, esquema y configuración por defecto (única fuente)
    ├── dataset.py       <- Descarga, validación, limpieza y huella de los datos
    ├── features.py      <- Separación X/y y partición train/validación/prueba
    ├── plots.py         <- Paleta y gráficas (compartidas con el notebook)
    ├── tracking.py      <- Registro en MLflow: parámetros, métricas, artefactos y modelo
    └── modeling
        ├── evaluate.py  <- Umbral de decisión y métricas de evaluación
        └── train.py     <- Punto de entrada (CLI) y orquestación del entrenamiento
```
