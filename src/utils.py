"""Utility functions shared across the GraphRAG project."""

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np


def set_random_seeds(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger and return a named logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("graphrag")


def create_project_dirs(base_dir: str = ".") -> None:
    """Create the standard directory layout under base_dir."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/outputs",
        "figures",
        "results",
        "notebooks",
        "src",
        "scripts",
        "configs",
    ]
    for d in dirs:
        Path(base_dir, d).mkdir(parents=True, exist_ok=True)
    print(f"Project directories ready under: {os.path.abspath(base_dir)}")


def save_dataframe(df, filename: str, results_dir: str = "results") -> str:
    """Save a DataFrame to CSV and return the file path."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(results_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_figure(fig, filename: str, figures_dir: str = "figures", dpi: int = 150) -> str:
    """Save a Matplotlib figure and return the file path."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")
    return path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load a YAML configuration file and return it as a dict."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
