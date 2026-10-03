"""Small reusable plotting helpers for the analysis and dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_scatter_with_trend(
    data: pd.DataFrame, x: str, y: str, path: Path, title: str
) -> None:
    sample = data[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample[x], sample[y], alpha=0.35, s=18)
    if len(sample) > 1 and sample[x].nunique() > 1:
        slope, intercept = np.polyfit(sample[x], sample[y], 1)
        xs = np.linspace(sample[x].min(), sample[x].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#d62728", linewidth=2)
    ax.set(title=title, xlabel=x, ylabel=y)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_correlation_heatmap(data: pd.DataFrame, columns: list[str], path: Path) -> None:
    corr = data[columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Selected feature correlation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
