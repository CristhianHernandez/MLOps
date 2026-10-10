"""Estilo visual compartido y gráficas de evaluación del modelo."""

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from diabetes_mlops_v2.config import FIGURES_DIR

# Paleta consistente para el notebook de exploración y los reportes del modelo
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#52514e"
PALETA_CLASE = {0: AZUL, 1: NARANJA}
ETIQUETA_CLASE = {0: "Sin diabetes (0)", 1: "Prediabetes/diabetes (1)"}
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#104281", AZUL, "#f0efec", "#e34948", "#9b1c1c"]
)
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)


def configurar_estilo() -> None:
    """Estilo general de las gráficas."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "grid.color": "#e8e7e3",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "figure.dpi": 100,
        }
    )


def graficar_matriz_confusion(y: pd.Series, predicciones: np.ndarray, umbral: float) -> plt.Figure:
    """Matriz de confusión con conteos y porcentaje por clase real."""
    matriz = confusion_matrix(y, predicciones)
    pct_fila = matriz / matriz.sum(axis=1, keepdims=True) * 100
    etiquetas = np.array(
        [
            [f"{matriz[i, j]:,}\n({pct_fila[i, j]:.1f}% de la fila)" for j in range(2)]
            for i in range(2)
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        pct_fila,
        annot=etiquetas,
        fmt="",
        cmap=CMAP_SECUENCIAL,
        vmin=0,
        vmax=100,
        linewidths=2,
        linecolor="white",
        cbar=False,
        annot_kws={"size": 11},
        ax=ax,
    )
    ax.set_xticklabels(["Predice: sin diabetes", "Predice: diabetes"])
    ax.set_yticklabels(["Real: sin diabetes", "Real: diabetes"], rotation=0)
    ax.set_title(f"Matriz de confusión (umbral = {umbral:.2f})")
    fig.tight_layout()
    return fig


def graficar_curvas_roc_pr(
    y: pd.Series, probabilidades: np.ndarray, metricas: dict, umbral: float
) -> plt.Figure:
    """Curva ROC y curva precisión-recall, marcando el punto del umbral elegido."""
    fpr, tpr, _ = roc_curve(y, probabilidades)
    precision, recall, _ = precision_recall_curve(y, probabilidades)
    etiqueta_umbral = f"Umbral elegido ({umbral:.2f})"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(fpr, tpr, color=NARANJA, lw=2, label=f"Modelo (AUC = {metricas['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color=GRIS, lw=1, ls="--", label="Azar (AUC = 0.5)")
    axes[0].scatter(
        1 - metricas["especificidad"],
        metricas["recall"],
        color=GRIS,
        s=70,
        zorder=3,
        label=etiqueta_umbral,
    )
    axes[0].set(
        title="Curva ROC",
        xlabel="Tasa de falsos positivos (1 − especificidad)",
        ylabel="Recall (tasa de verdaderos positivos)",
    )
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(
        recall, precision, color=NARANJA, lw=2, label=f"Modelo (PR-AUC = {metricas['pr_auc']:.3f})"
    )
    axes[1].axhline(
        y.mean(), color=GRIS, lw=1, ls="--", label=f"Azar (= prevalencia, {y.mean():.3f})"
    )
    axes[1].scatter(
        metricas["recall"],
        metricas["precision"],
        color=GRIS,
        s=70,
        zorder=3,
        label=etiqueta_umbral,
    )
    axes[1].set(
        title="Curva precisión-recall", xlabel="Recall", ylabel="Precisión", ylim=(0, 1.02)
    )
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def graficar_coeficientes(coeficientes: pd.Series) -> plt.Figure:
    """Coeficientes del modelo: naranja aumenta el riesgo, azul lo reduce."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        coeficientes.index,
        coeficientes.values,
        height=0.6,
        color=[NARANJA if v > 0 else AZUL for v in coeficientes.values],
    )
    for i, v in enumerate(coeficientes.values):
        ax.text(
            v + (0.02 if v >= 0 else -0.02),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=GRIS,
        )
    ax.axvline(0, color=GRIS, lw=1)
    ax.set_xlim(coeficientes.min() - 0.25, coeficientes.max() + 0.25)
    ax.set(
        title="Coeficientes de la regresión logística (variables estandarizadas)",
        xlabel="Coeficiente (cambio en log-odds por 1 desviación estándar)",
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def guardar_figura(fig: plt.Figure, nombre: str, directorio: Path = FIGURES_DIR) -> Path:
    """Guarda la figura como PNG y libera la memoria."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{nombre}.png"
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return ruta
