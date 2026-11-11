"""Probabilistic metrics and plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src import config

EPS = 1e-15


def one_hot(y: np.ndarray, k: int = 3) -> np.ndarray:
    return np.eye(k)[np.asarray(y, dtype=int)]


def log_loss(y, P) -> float:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), np.asarray(y, dtype=int)])))


def brier(y, P) -> float:
    return float(np.mean(np.sum((np.asarray(P) - one_hot(y)) ** 2, axis=1)))


def rps(y, P) -> float:
    """Ranked Probability Score (lower is better). Outcomes are ordered H < D < A."""
    cp = np.cumsum(np.asarray(P), axis=1)[:, :-1]
    co = np.cumsum(one_hot(y), axis=1)[:, :-1]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2.0))


def accuracy(y, P) -> float:
    return float(np.mean(np.argmax(P, axis=1) == np.asarray(y, dtype=int)))


def summarize(name: str, y, P) -> dict:
    return {"model": name, "log_loss": log_loss(y, P), "brier": brier(y, P),
            "rps": rps(y, P), "accuracy": accuracy(y, P), "n": len(y)}


# ---------------------------------------------------------------- plots
def plot_model_comparison(results, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["model"] for r in results]
    vals = [r["log_loss"] for r in results]
    bars = ax.barh(names, vals, color="#37003c")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Log loss on test seasons (lower is better)")
    ax.set_xlim(min(vals) * 0.95, max(vals) * 1.03)
    ax.invert_yaxis()
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y, P, path, title="Calibration") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for k, (ax, label) in enumerate(zip(axes, ["Home win", "Draw", "Away win"])):
        frac, mean_pred = calibration_curve(one_hot(y)[:, k], P[:, k], n_bins=8,
                                            strategy="quantile")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(mean_pred, frac, "o-", color="#e90052")
        ax.set_title(label)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, P, path, title="Confusion matrix") -> None:
    cm = confusion_matrix(y, np.argmax(P, axis=1), labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(3), config.CLASSES)
    ax.set_yticks(range(3), config.CLASSES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(names, values, path, top: int = 15) -> None:
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#00ff85",
            edgecolor="#37003c")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
