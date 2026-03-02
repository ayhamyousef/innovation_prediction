"""
Utility helpers: config loading, reproducibility, logging, device setup.
"""

import os
import random
import logging
import yaml
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def load_config(path: str = "config/default.yaml", overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """Load YAML config with optional dict overrides."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, overrides)
    # Resolve API key from env if not in config
    if cfg.get("data", {}).get("api_key") is None:
        cfg["data"]["api_key"] = os.environ.get("ODP_API_KEY", "")
    return cfg


def _deep_update(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to file + console."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"run_{timestamp}.log")

    logger = logging.getLogger("innovation_prediction")
    logger.setLevel(level)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_json(data: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
